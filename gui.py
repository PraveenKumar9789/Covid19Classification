if True:
    from utils import reset_random
    reset_random()

import os
from cv2 import cv2
from PyQt5.QtWidgets import (QDialog, QProgressBar, QApplication, QWidget, QVBoxLayout, QGroupBox,
                             QGridLayout, QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFontDatabase, QFont, QImage, QPixmap
from utils import CLASSES
from feature_extractor import resnet_152, extract_feature
from segmenter import segment_image, get_segmented_data, load_model, dice_coef_loss, dice_coef
import pickle
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot

import traceback
import sys


class WeedDetectionClassification(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CovidDetectionClassification')
        self.screen_size = app.primaryScreen().availableSize()
        self.app_width = (self.screen_size.width() // 100) * 100
        self.app_height = (self.screen_size.height() // 100) * 99
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setGeometry(0, 0, self.app_width, self.app_height)

        QFontDatabase.addApplicationFont('assets/FiraCode-Retina.ttf')
        app.setFont(QFont('Fira Code Retina', 12))

        self.full_v_box = QVBoxLayout()
        self.top_h_box = QVBoxLayout()
        self.bottom_h_box = QVBoxLayout()
        self.full_v_box.addLayout(self.top_h_box)
        self.full_v_box.addLayout(self.bottom_h_box)

        self.gb_1 = QGroupBox('Input Data')
        self.gb_2 = QGroupBox('Results')

        self.gb_1.setFixedWidth(self.app_width)
        self.gb_1.setFixedHeight((self.app_height // 100) * 25)
        self.grid_1 = QGridLayout()
        self.grid_1.setSpacing(10)
        self.gb_1.setLayout(self.grid_1)

        self.ip_le = QLineEdit()
        self.ip_le.setFixedWidth((self.app_width // 100) * 35)
        self.ip_le.setFocusPolicy(Qt.NoFocus)
        self.grid_1.addWidget(self.ip_le, 0, 0)

        self.ci_pb = QPushButton('Choose X-Ray Image')
        self.ci_pb.clicked.connect(self.choose_input)
        self.grid_1.addWidget(self.ci_pb, 1, 0)

        self.seg_pb = QPushButton('Segment Lung Region')
        self.seg_pb.clicked.connect(self.seg_thread)
        self.grid_1.addWidget(self.seg_pb, 0, 1)

        self.fe_pb = QPushButton('Resnet-152 Feature Extraction')
        self.fe_pb.clicked.connect(self.feature_extract_thread)
        self.grid_1.addWidget(self.fe_pb, 1, 1)

        self.cls_pb = QPushButton('RandomForest Classification')
        self.cls_pb.clicked.connect(self.classify_thread)
        self.grid_1.addWidget(self.cls_pb, 0, 2)

        self.re_pb = QPushButton('Reset')
        self.re_pb.clicked.connect(self.reset)
        self.grid_1.addWidget(self.re_pb, 1, 2)

        self.gb_2.setFixedHeight((self.app_height // 100) * 80)
        self.grid_2_scroll = QScrollArea()
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.grid_2 = QGridLayout(self.grid_2_widget)
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)
        self.gb_2.setLayout(self.gb_2_v_box)

        self.top_h_box.addWidget(self.gb_1)
        self.bottom_h_box.addWidget(self.gb_2)
        self.setLayout(self.full_v_box)

        self.input_image_path = ''
        self.image_size = ((self.gb_2.height() // 100) * 90, (self.app_width // 100) * 45)
        self.load_screen = Loading()
        self.thread_pool = QThreadPool()
        self.classifier = pickle.load(open('rf_model.pkl', 'rb'))
        self.detection_dict = {}
        self.feature = None
        self.class_ = None
        self.disable()
        self.index = 0
        self.mdl = resnet_152()
        self.unet_mdl = load_model('unet-model/model.h5',
                                   custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
        self.show()

    def add_image(self, im_path, title):
        image_lb = QLabel()
        image_lb.setFixedHeight(self.image_size[0])
        image_lb.setFixedWidth(self.image_size[1])
        image_lb.setScaledContents(True)
        image_lb.setStyleSheet('padding-top: 30px;')
        qimg = QImage(im_path)
        pixmap = QPixmap.fromImage(qimg)
        image_lb.setPixmap(pixmap)
        self.grid_2.addWidget(image_lb, 0, self.index, Qt.AlignCenter)
        txt_lb = QLabel(title)
        self.grid_2.addWidget(txt_lb, 1, self.index, Qt.AlignCenter)
        self.index += 1

    @staticmethod
    def show_message_box(title, icon, msg):
        msg_box = QMessageBox()
        msg_box.setFont(QFont('FiraCode', 10, 1))
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(icon)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.exec_()

    def choose_input(self):
        self.reset()
        self.input_image_path, _ = QFileDialog.getOpenFileName(self,
                                                               caption="Choose Input Image", directory=".",
                                                               options=QFileDialog.DontUseNativeDialog,
                                                               filter="JPG Files (*.jpg);;BMP Files (*.bmp);;"
                                                                      "PNG Files (*.png)")
        if os.path.isfile(self.input_image_path):
            self.ip_le.setText(self.input_image_path)
            self.add_image(self.input_image_path, 'Input Image')
            self.seg_pb.setEnabled(True)
            self.ci_pb.setEnabled(False)
        else:
            self.show_message_box('InputImageError', QMessageBox.Critical, 'Choose valid image?')

    @staticmethod
    def clear_layout(layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                continue
            w = item.widget()
            if w:
                w.deleteLater()

    def seg_thread(self):
        worker = Worker(self.seg_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.seg_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()

    def seg_runner(self, progress_callback):
        image_, mask_ = segment_image(self.input_image_path, self.unet_mdl)
        mask_, segmented_, drawn_ = get_segmented_data(image_, mask_)
        self.detection_dict['Mask Detected'] = mask_
        self.detection_dict['Mask Drawn'] = drawn_
        self.detection_dict['Mask Segmented'] = segmented_
        progress_callback.emit('Done')

    def seg_finisher(self):
        for k, v in self.detection_dict.items():
            cv2.imwrite(k+'.jpg', v)
            self.add_image(k+'.jpg', k)
        self.fe_pb.setEnabled(True)
        self.seg_pb.setEnabled(False)
        self.load_screen.close()

    def feature_extract_thread(self):
        worker = Worker(self.feature_extract_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.feature_extract_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()

    def feature_extract_runner(self, progress_callback):
        self.feature = extract_feature('Mask Segmented.jpg', self.mdl)
        progress_callback.emit('Done')

    def feature_extract_finisher(self):
        self.fe_pb.setEnabled(False)
        self.cls_pb.setEnabled(True)
        self.load_screen.close()

    def classify_thread(self):
        worker = Worker(self.classify_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.classify_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()

    def classify_runner(self, progress_callback):
        reset_random()
        prob = np.argmax(self.classifier.predict_proba(np.array(self.feature, ndmin=2)), axis=1)[0]
        pred = self.classifier.predict(np.array(self.feature, ndmin=2))[0]
        self.class_ = (round(prob * 100, 2), pred)
        progress_callback.emit('Done')

    def classify_finisher(self):
        self.add_image(self.input_image_path, 'Classified As :: {0}'.format(CLASSES[self.class_[1]]))
        self.cls_pb.setEnabled(False)
        self.ci_pb.setEnabled(True)
        self.load_screen.close()

    def disable(self):
        self.ip_le.clear()
        self.ci_pb.setEnabled(True)
        self.input_image_path = ''
        self.seg_pb.setEnabled(False)
        self.fe_pb.setEnabled(False)
        self.cls_pb.setEnabled(False)
        self.detection_dict = {}
        self.feature = None
        self.class_ = None
        self.index = 0

    def reset(self):
        self.disable()
        self.clear_layout(self.grid_2)


class Loading(QDialog):
    def __init__(self, parent=None):
        super(Loading, self).__init__(parent)
        self.screen_size = app.primaryScreen().size()
        self._width = int(self.screen_size.width() / 100) * 40
        self._height = int(self.screen_size.height() / 100) * 5
        self.setGeometry(0, 0, self._width, self._height)
        x = (self.screen_size.width() - self.width()) / 2
        y = (self.screen_size.height() - self.height()) / 2
        self.move(x, y)
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.pb = QProgressBar(self)
        self.pb.setFixedWidth(self.width())
        self.pb.setFixedHeight(self.height())
        self.pb.setRange(0, 0)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as err:
            print(err)
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    th = WeedDetectionClassification()
    sys.exit(app.exec_())
