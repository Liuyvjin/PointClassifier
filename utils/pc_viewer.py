import sys
import os.path as osp
BASE_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(BASE_DIR, '..'))
import numpy as np
from utils.pc_util import rot_angle_axis, read_ply
from utils.pc_render import PC_Render
from pointfield.loss_utils import chamfer_pc
from PyQt5.QtWidgets import ( QStatusBar, QVBoxLayout, QWidget,
                             QLabel, QApplication)
from PyQt5 import QtGui as QG
from PyQt5.QtCore import Qt
from time import sleep
from PIL import ImageQt
import torch


def load_shape_name():
        shape_file = osp.join(BASE_DIR, '..\\data\\modelnet40_ply_hdf5_2048\\shape_names.txt')
        with open(shape_file, 'r') as fd:
            lines = fd.readlines()
            shape_name = [line.strip() for line in lines]
        return shape_name

def idx_generator(size):
    idx = 0
    while True:
        yield idx
        idx = 0 if idx>=size-1 else idx+1

class PC_Viewer(QWidget):
    """
    Point cloud visualization Gui.
    """
    def __init__(   self,
                    pointcloud,
                    rgb         =   [99, 184, 255],
                    alpha       =   1,
                    diameter    =   25,
                    bg_color    =   [255, 255, 255] ,
                    dataset     =   None):

        super().__init__()
        # 数据集
        self.shape_name = load_shape_name()
        self.dataset = dataset
        self.idx_iter = idx_generator(len(dataset))
        self.pc1 = pointcloud  # 初始点云
        self.label = 0 # 按n后为当前点云 label
        # 渲染
        self.pc_rander = PC_Render(
            pointcloud = self.pc1,
            rgb = rgb,
            alpha = alpha,
            bg_color = bg_color,
            diameter = diameter,
            canvas_size = 700,
            paint_size = 300
        )
        self.bg_color = bg_color
        self.diameter = diameter
        self.canva_size = 700
        self.paint_size = 300
        # 鼠标起始位置
        self.x_start = 0
        self.y_start = 0
        # 变换
        self.rot_start = np.eye(3)
        self.rot_relative = np.eye(3)
        self.scale = 1  # paint scale factor

        self.initUI()
        self.update_img()
        self.update_msg = self.status_bar.showMessage


    def initUI(self):
        # background color
        pal = QG.QPalette()
        pal.setColor(self.backgroundRole(), QG.QColor(*self.bg_color))
        self.setPalette(pal)
        # widgets
        self.pc_viewer = QLabel(self)
        self.pc_viewer.setMinimumSize(50, 50)
        self.status_bar = QStatusBar(self)
        self.status_bar.setFixedHeight(20)
        # layout
        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.addWidget(self.pc_viewer)
        v_layout.addWidget(self.status_bar)
        self.setLayout(v_layout)

        self.pc_viewer.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(False)

        self.move(300, 100)
        self.resize(700, 720)
        self.setWindowTitle('Pointcloud Viewer')
        self.show()


    def mousePressEvent(self, pos) -> None:
        """ save mouse press position """
        if pos.y() > self.height()-20:
            return
        self.x_start = pos.x()
        self.y_start = pos.y()
        self.rot_start = np.dot(self.rot_relative, self.rot_start)

    def mouseMoveEvent(self, pos):
        """ rotate point cloud """
        if pos.y() > self.height()-20:
            return
        self.update_msg('x={:d}, y={:d}'.format(pos.x(),  pos.y()))

        dx = pos.x() - self.x_start
        dy = pos.y() - self.y_start
        axis = np.array([dx, -dy, 0])
        angle = np.sqrt(dx**2 + dy**2) / 200
        self.rot_relative = rot_angle_axis(angle, axis)
        self.update_img()

    def wheelEvent(self, event: QG.QWheelEvent) -> None:
        """ scale point cloud """
        if event.angleDelta().y()>0:
            self.scale += 0.15
        else:
            self.scale -= 0.15
        self.scale = max(self.scale, 0.3)
        self.update_img()


    def update_img(self):
        """ redraw image """
        diameter = int(self.scale * self.diameter)
        paint_size = int(self.scale * self.paint_size)
        rot = np.dot(self.rot_relative, self.rot_start)
        self.pc_rander.update(diameter=diameter, paint_size=paint_size, rot=rot)
        img = self.pc_rander.draw()
        self.img = QG.QPixmap(QG.QImage(img.data, img.shape[1], img.shape[0],
                            img.shape[1]*3, QG.QImage.Format_RGB888))
        self.pc_viewer.setPixmap(self.img)


    def resizeEvent(self, a0: QG.QResizeEvent) -> None:
        self.pc_rander.update(canvas_size=[self.pc_viewer.height(), self.pc_viewer.width()])

    def keyPressEvent(self, keyevent):
        """ 按键处理
            - n : 加载下一张图
            - s : 加载下一张同类图片
            - 1 : 更新pc1
            - c : 计算pc1和当前pc的chamfer distance
            - ESC : 退出
        """
        key = keyevent.text()
        if f'{keyevent.key():X}' == '1000000':  # ESC
            self.close()
        if key == 'n' and self.dataset is not None:  # next
            idx = next(self.idx_iter)
            pc, label = self.dataset[idx]
            print(f"key: {keyevent.text()} | idx: {idx:>4d} | label: {self.shape_name[label[0]]:>8s}")
            self.pc_rander.pc = pc
            self.label = label
            self.update_img()
        if key == 's' and self.dataset is not None:  # next same item
            while True:
                idx = next(self.idx_iter)
                pc, label = self.dataset[idx]
                if label == self.label :
                    break
            print(f"key: {keyevent.text()} | idx: {idx:>4d} | label: {self.shape_name[label[0]]:>8s}")
            self.pc_rander.pc = pc
            self.update_img()
        if key == '1':
            self.pc1 = self.pc_rander.pc
            print('update pc1')
        if key == 'c':
            dist = chamfer_pc(torch.Tensor(self.pc1)[None], torch.Tensor(self.pc_rander.pc)[None], 1024)[0]
            print(f"chamfer distance: {dist}")


    def rotate(self, filename, save=False):
        images = []
        axis = np.array([1, -0.2, 0])
        # self.rot_start = rot_angle_axis(np.pi/2, np.array([0, 0, 1]))

        for i in range(30):
            angle = np.pi * i / 15
            self.rot_relative = rot_angle_axis(angle, axis)
            self.update_img()
            QApplication.processEvents()
            if save:
                images.append(ImageQt.fromqpixmap(self.img))
            sleep(0.02)
        if save:
            images[0].save(osp.join(BASE_DIR, '../data/shape_visual', filename+'.gif'),
                save_all=True, append_images=images[1:], optimize=True, duration=80, loop=0)


if __name__=="__main__":
    from data.data_utils import ModelNet40



    app = QApplication(sys.argv)

    # --- dataset
    dataset = ModelNet40(partition='test', num_points=1024)
    print(f'Dataset size: {len(dataset)}')
    shape_name = load_shape_name()

    # --- 可视化
    pointcloud, label = dataset[0]
    exp = PC_Viewer(pointcloud, diameter=15, dataset=dataset)

    sys.exit(app.exec_())

    # --- 为每个种类别生成动图
    # labels = set()
    # for i in range(30):
    #     pointcloud, label = dataset[i]
    #     label = label[0]
    #     if label in labels:
    #         continue
    #     else:
    #         labels.add(label)
    #         exp = PC_Viewer(pointcloud, diameter=15)
    #         # exp.rotate(shape_name[label])