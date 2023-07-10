import sys
from distutils.dir_util import copy_tree
from PyQt5.QtCore import QProcess
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
import numpy as np
import cv2
import tensorflow as tf

form_class = uic.loadUiType("cnn_pj.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

#   ----------------------------↓↓↓변수↓↓↓-----------------------------------

        # 모델
        self.model_path = ""

        # 데이터셋 파일 저장 리스트
        self.dataset_list_dir = ""

        # 데이터셋 적용폴더 저장 리스트
        self.dataset_dir = ""

        # 결과폴더 절대경로
        self.result_dir = ""

        # 텍스트 데이터 모든파일의 절대경로
        self.text_all_list = []

        # 그림 데이터 모든파일의 절대경로
        self.img_all_list = []

        # 화면에 표시된 데이터셋 파일과 결과파일의 절대값 주소
        self.original_img = ""
        self.result_img_img = ""

        # treeView 목록 파일을 클릭하면 그 파일의 절대값을 저장
        self.tree_View_file_click = ""

        # 마진조절
        self.margin = 15

        # area조절
        self.area = 1000

#   ----------------------------↑↑↑변수↑↑↑-----------------------------------

#   ----------------------------↓↓↓버튼↓↓↓-----------------------------------

        # 모델/데이터셋 불러오기 버튼
        self.pushButton_2.clicked.connect(self.datasetDirSelect)
        # 결과폴더 선택버튼
        self.pushButton_7.clicked.connect(self.resultDir)
        # 리셋버튼
        self.pushButton_5.clicked.connect(self.reset)
        # 마진조정버튼
        self.pushButton_margin.clicked.connect(self.marginC)
        # area조정버튼
        self.pushButton_area.clicked.connect(self.areaC)
        # 결과확인 버튼 test
        self.pushButton_result.clicked.connect(self.numberDetection)

#   ----------------------------↑↑↑버튼↑↑↑-----------------------------------

#   ----------------------------↓↓↓함수들↓↓↓--- --------------------------------

    # 모델위치 선택 함수
    def modelSelect(self):
        file_path = QFileDialog.getOpenFileName(self, 'Attach File')[0]
        self.model_path = file_path

    # 데이터셋이 들어있는 폴더를 선택하는 함수
    def datasetDirSelect(self):
        file_path = QFileDialog.getExistingDirectory(self, "select Directory")
        self.dataset_list_dir = file_path
        self.model_file_system = QFileSystemModel()
        self.model_file_system.setRootPath(file_path)
        self.model_file_system.setReadOnly(False)
        self.treeView.setModel(self.model_file_system)
        self.treeView.setRootIndex(self.model_file_system.index(file_path))
        self.treeView.doubleClicked.connect(lambda index: self.treeViewDoubleClicked1(index))
        self.treeView.setDragEnabled(True)
        self.treeView.setColumnWidth(0, 300)

    # treeView 더블클릭 시 절대값 추출함수
    def treeViewDoubleClicked1(self, index):
        self.tree_View_file_click = self.model_file_system.filePath(index)
        self.original_img = self.tree_View_file_click
        pixmap = QPixmap(self.tree_View_file_click)
        pixmap = pixmap.scaled(self.label.size(), aspectRatioMode=True)
        self.label.setPixmap(pixmap)

    def treeViewDoubleClicked2(self, index):
        self.tree_View_file_click = self.model_file_system.filePath(index)
        self.result_img = self.tree_View_file_click
        pixmap = QPixmap(self.tree_View_file_click)
        pixmap = pixmap.scaled(self.label_2.size(), aspectRatioMode=True)
        self.label_2.setPixmap(pixmap)

    # 실행버튼 함수
    def runFile(self):
        copy_tree(self.dataset_list_dir, self.dataset_dir)
        file_path = self.model_path
        process = QProcess(self)
        process.start('python', [file_path])

    # 데이터셋을 적용할 폴더 선택 함수
    def datasetDir(self):
        directory_path = QFileDialog.getExistingDirectory(self, "select Directory")
        self.dataset_dir = directory_path

    # 결과폴더 선택
    def resultDir(self):
        file_path = QFileDialog.getExistingDirectory(self, "select Directory")
        self.result_dir = file_path
        print(self.result_dir)
        self.model_file_system = QFileSystemModel()
        self.model_file_system.setRootPath(file_path)
        self.model_file_system.setReadOnly(False)
        self.treeView_2.setModel(self.model_file_system)
        self.treeView_2.setRootIndex(self.model_file_system.index(file_path))
        self.treeView_2.doubleClicked.connect(lambda index: self.treeViewDoubleClicked2(index))
        self.treeView_2.setDragEnabled(True)
        self.treeView_2.setColumnWidth(0, 300)

    def reset(self):
        self.label.clear()
        self.label_2.clear()
        self.lineEdit_area.clear()
        self.lineEdit_margin.clear()
        self.margin = 15
        self.area = 1000

    def marginC(self):
        text = self.lineEdit_margin.text()
        self.margin = text

    def areaC(self):
        text = self.lineEdit_area.text()
        self.area = text

    def numberDetection(self):
        src = cv2.imread(self.original_img)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.dilate(gray, kernel, iterations=1)

        ret_val, binImg = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
        ret_val, gray = cv2.threshold(gray, 94, 255, cv2.THRESH_TOZERO)
        n_blob, labelImg, stats, centroid = cv2.connectedComponentsWithStats(binImg)
        model = tf.keras.models.load_model("cnn_model.h5")

        margin = self.margin
        for i in range(1, n_blob):
            x, y, w, h, area = stats[i]

            if area > (self.area):
                x -= margin
                y -= margin
                w += margin * 2
                h += margin * 2
                cv2.rectangle(src, (x, y, w, h), (255, 0, 255), thickness=2)

                crop = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                crop = gray[y:y + h, x:x + w].copy()

                crop = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)

                input_x = np.expand_dims(crop, axis=0)
                output_y = model.predict(input_x)

                ans = output_y.argmax()
                conf = output_y[0, ans]

                show_str = str(ans)
                print(show_str)
                cv2.putText(src, show_str, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                cv2.imwrite(f'{self.result_dir}/{i - 1}_result{show_str}.png', crop)
                text = open(f'{self.result_dir}/text_result', 'w')
                text.write(f'{show_str}\n')

        # cv2.imshow("crop ing", crop)
        # cv2.imshow("gray img", gray)
        # cv2.imshow("original img", src)
        # cv2.imshow("binary img", binImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



#   ----------------------------↑↑↑함수들↑↑↑-----------------------------------

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
