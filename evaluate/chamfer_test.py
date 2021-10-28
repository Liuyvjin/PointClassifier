import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.joinpath('..').resolve()))
from PyQt5.QtWidgets import ( QStatusBar, QVBoxLayout, QWidget,
                             QLabel, QApplication)

from utils.pc_viewer import PC_Viewer
from pointfield.loss_utils import chamfer_pc
from data.data_utils import ModelNet40


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # dataset
    dataset = ModelNet40(partition='test', num_points=1024)

    # --- 可视化
    pointcloud, label = dataset[0]
    exp = PC_Viewer(pointcloud, diameter=15, dataset=dataset)

    sys.exit(app.exec_())

