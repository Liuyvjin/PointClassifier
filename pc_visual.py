import sys
import os.path as osp
from data.data_utils import ModelNet40
from utils.pc_util import draw_pointcloud_rgb, rot_angle_axis
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QMainWindow, QStatusBar, QWidget, QHBoxLayout,
                             QLabel, QApplication)
from PyQt5.QtGui import QImage, QPixmap
from pointfield import PointField
import torch

BASE_DIR = osp.dirname(osp.abspath(__file__))


class PCVisual(QMainWindow):

    def __init__(self, pointcloud):
        super().__init__()
        self.pc = pointcloud
        self.rgb = [99 ,184, 255]
        self.x0 = 0
        self.y0 = 0
        self.rot90 = np.array( [[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])
        self.rot_start = np.eye(3)
        self.rot_relative = np.eye(3)
        self.update_img(self.rot_start)
        self.initUI()

    def initUI(self):
        self.viewer = QLabel(self)
        self.viewer.setPixmap(QPixmap(self.img))
        self.setCentralWidget(self.viewer)

        self.show_msg = self.statusBar().showMessage

        self.setMouseTracking(False)

        self.move(300, 100)
        self.setWindowTitle('Pointcloud Visualization')
        self.show()

    def mousePressEvent(self, pos) -> None:
        self.x0 = pos.x()
        self.y0 = pos.y()
        self.rot_start = np.dot(self.rot_relative, self.rot_start)

    def mouseMoveEvent(self, pos):
        self.show_msg('x={:d}, y={:d}'.format(pos.x(),  pos.y()))
        dx = pos.x() - self.x0
        dy = pos.y() - self.y0

        axis = np.dot(self.rot90, np.array([[dy],[dx],[0]])).reshape(3)
        angle = np.sqrt(dx**2 + dy**2) / 200
        self.rot_relative = rot_angle_axis(angle, axis)
        rot = np.dot(self.rot_relative, self.rot_start)
        self.update_img(rot)
        self.viewer.setPixmap(QPixmap(self.img))

    def update_img(self, rot):
        img = draw_pointcloud_rgb(self.pc, alpha=1 , rgb=self.rgb,
                            canvasSize=700, space=300, diameter=30, rot=rot)
        self.img = QImage(img.data, img.shape[1], img.shape[0],
                            img.shape[1]*3, QImage.Format_RGB888)


if __name__=="__main__":
    app = QApplication(sys.argv)

    data = ModelNet40(500, partition='test')
    pointcloud = data[466][0]

    # pointfield
    pf_path = BASE_DIR + '\\logs\\pointfield_margin03_dgcnn\\checkpoints\\best_pointfield.t7'
    pf = PointField(64)
    pf.load_state_dict(torch.load(pf_path)['model_state_dict'])
    pc = torch.Tensor(pointcloud)[None,...].cuda()
    # print(pf.shape)
    pc = pf(pc)
    pc = pc.squeeze().detach().cpu().numpy()
    ex1 = PCVisual(pointcloud)
    ex = PCVisual(pc)
    sys.exit(app.exec_())

