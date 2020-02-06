# import keperluan program
# ---------------------------------------------------------------------------------------------

# untuk keperluan file management
import os

# untuk keperluan keberjalanan program
import sys

# untuk sistem ekstraksi fitur
import cv2

# untuk operasi matematis yang efisien
import math

# untuk penyimpanan database
import pickle

# untuk struktur data dan operasi matematis yang efisien
import numpy as np

# untuk loading bar saat membuat database
from tqdm import tqdm

# untuk sistem pembacaan gambar
from imageio import imread

# untuk sistem antarmuka pengguna grafis
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

# konstanta / setting program
# ---------------------------------------------------------------------------------------------

DATABASE_NAME = 'features.db'
FOLDER_NAME = 'PINS'

# fungsi-fungsi umum
# ---------------------------------------------------------------------------------------------

# pengondisian alamat file supaya program dapat berjalan baik di Windows maupun di Linux
def fixslashes(path):
    if (os.sep == '/'):
        if '\\' in path:
            return path.replace('\\', os.sep)
    else:
        if '/' in path:
            return path.replace('/', os.sep)
    return path

# menghitung panjang vektor
def norm (a, use_numpy=False):
    if (use_numpy):
        return np.linalg.norm(a)
    else:
        ret = 0
        for i in range (len(a)):
            ret += a[i] * a[i]
        return math.sqrt(ret)

# menghitung panjang dua vektor
def dpsNorm (a, b, use_numpy=False):
    if (use_numpy):
        return np.linalg.norm(b-a)
    else:
        ret = 0
        for i in range (len(a)):
            ret += (a[i]-b[i]) * (a[i]-b[i])
        return math.sqrt(ret)

# menghitung dot product dua vektor
def dotProduct(a,b, use_numpy=False):
    if (use_numpy):
        return np.dot(a,b)
    else:
        ret = 0
        for i in range (len(a)):
            ret += a[i] * b[i]
        return ret

# menghitung nilai kosinus dari dua vektor
def angleFromDotProduct (a,b,c, use_numpy=False): 
    angle = dotProduct(a,b, use_numpy)/(norm(a, use_numpy)*c)
    return angle

# mengekstrak fitur dari gambar pada suatu alamat
def extract_features(image_path, vector_size=32):
    image = imread(image_path, pilmode="RGB")
    try:
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key = lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        if (len(kps) < 1 or dsc is None):
            print('\nFailed to read from %s.' % image_path)
            dsc = np.zeros(vector_size * 64)
        else:
            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print("Error", e)
        return None
    return dsc

# menulis semua file yang ada di alamat folder database
def list_file(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

# mencari list berisi fitur dari semua gambar pada suatu list alamat file gambar
def extract_all_data(files):
    features = []
    for i in tqdm(files):
        features.append(extract_features(i))
    return features

# mencari gambar yang sudah ditandai dengan keypoint jika fitur penampilan keypoint diaktifkan
def get_keypointed_image(path, go=True, vector_size=32):
    if (go):
        image = imread(path, pilmode="RGB")
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key = lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.drawKeypoints(image, kps, outImage=np.array([]))
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    else:
        return path

# program utama
# ---------------------------------------------------------------------------------------------

# kelas program utama
class App(QWidget):

    # variabel global program
    # ---------------------------------------------------------------------------------------------

    # database dan informasi mengenai database
    db = {}
    db['files'] = []
    db['features'] = []
    dbSign = '/'

    # list hasil matching untuk ditampilkan
    currentList = []

    # alamat file database
    featuresPath = DATABASE_NAME

    # memperbarui tampilan hasil matching
    def updateList(self):
        self.resultList.clear()

        c = self.controllerSpinner.value()
        if (len(self.currentList) < c):
            c = len(self.currentList)

        pdialog = QProgressDialog("Menampilkan gambar hasil...", "Cancel", 0, c, self)
        pdialog.setWindowTitle('Drawing...')
        pdialog.setMinimumDuration(500)
        pdialog.setWindowModality(Qt.WindowModal)
        pdialog.setValue(0)
        pdialog.show()
        for i in range(c):
            item = QListWidgetItem()
            shower = QWidget()
            image = QLabel()
            pixmap = QPixmap(get_keypointed_image(fixslashes(self.currentList[i][1]), self.showKeypoints.checkState()))
            image.setPixmap(pixmap)
            image.setAlignment(Qt.AlignCenter)
            cl = self.currentList[i][1].split(self.dbSign)
            index = i+1
            if (self.reverse.checkState()):
                index = len(self.currentList)-i
            string = '[{j}]\n{fn}\n{fon}\n{:.5f}'.format(self.currentList[i][0], j = index, fn = cl[-1], fon = cl[-2])
            text = QLabel(string)
            text.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout()
            layout.addWidget(image)
            layout.addWidget(text)
            layout.addStretch()
            shower.setLayout(layout)
            shower.setFixedWidth(335)
            item.setSizeHint(shower.sizeHint())
            self.resultList.addItem(item)
            self.resultList.setItemWidget(item, shower)
            if pdialog.wasCanceled():
                break
            pdialog.setValue(pdialog.value()+1)
        pdialog.reset()
        pdialog.hide()

    # menampilkan tampilan pemilihan file gambar
    def pressLoad(self):
        self.path, _ = self.filepicker.getOpenFileName()
        if (self.path != ''):
            self.pathfinder.setText(self.path)
            self.pixmap = QPixmap(get_keypointed_image(self.path, self.showKeypoints.checkState()))
            self.label.setPixmap(self.pixmap)

    # melakukan matching untuk ditaruh pada currentList
    def compare(self):
        fitur_image_in = extract_features(self.path)
        self.currentList = [] 
        count = len(self.db['features'])
        pdialog = QProgressDialog("Membandingkan deskriptor gambar dengan semua deskriptor di database...", "Cancel", 0, count, self)
        pdialog.setWindowTitle('Comparing...')
        pdialog.setMinimumDuration(500)
        pdialog.setWindowModality(Qt.WindowModal)
        pdialog.setValue(0)
        pdialog.show()
        cosine = False
        if (self.methodOption.currentIndex() == 0):
            for i in range(count):
                self.currentList.append((dpsNorm(self.db['features'][i],fitur_image_in,True),self.db['files'][i]))
                if pdialog.wasCanceled():
                    break
                pdialog.setValue(pdialog.value()+1)
        elif (self.methodOption.currentIndex() == 1):
            cosine = True
            for i in range(count):
                self.currentList.append((angleFromDotProduct(self.db['features'][i], fitur_image_in,norm(fitur_image_in, True),True),self.db['files'][i]))
                if pdialog.wasCanceled():
                    break
                pdialog.setValue(pdialog.value()+1)
        elif (self.methodOption.currentIndex() == 2):
            for i in range(count):
                self.currentList.append((dpsNorm(self.db['features'][i],fitur_image_in),self.db['files'][i]))
                if pdialog.wasCanceled():
                    break
                pdialog.setValue(pdialog.value()+1)
        elif (self.methodOption.currentIndex() == 3):
            cosine = True
            for i in range(count):
                self.currentList.append((angleFromDotProduct(self.db['features'][i], fitur_image_in,norm(fitur_image_in)),self.db['files'][i]))
                if pdialog.wasCanceled():
                    break
                pdialog.setValue(pdialog.value()+1)
        pdialog.reset()
        pdialog.hide()
        rev = cosine
        if (self.reverse.checkState()):
            rev = not rev
        self.currentList = sorted(self.currentList,key=lambda i: i[0], reverse=rev)
        self.updateList()

    # konstruktor program utama
    def __init__(self):
        super().__init__()
        self.title = 'Autokenal Face Recognizer'
        self.left = 10
        self.top = 10
        self.width = 350
        self.height = 400
        self.initUI()
        if (os.path.exists(self.featuresPath)):
            print("Found a database to load at '%s'." % self.featuresPath)
            f = open(self.featuresPath, 'rb')
            self.db = pickle.load(f)
            sample = self.db['files'][0]
            if '/' in sample:
                self.dbSign = '/'
            else:
                self.dbSign = '\\'
            f.close()
        else:
            self.db['files'] = list_file(FOLDER_NAME)
            self.db['features'] = extract_all_data(self.db['files'])
            self.dbSign = os.sep
            f = open(self.featuresPath, 'wb')
            pickle.dump(self.db, f)
            f.close()
    
    # fungsi-fungsi inisialisasi antarmuka pengguna grafis
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Main Layout
        self.layout = QHBoxLayout(self)
        self.left = QVBoxLayout()
        self.left.setAlignment(Qt.AlignTop)
        self.right = QVBoxLayout()
        self.right.setAlignment(Qt.AlignTop)
        self.layout.addLayout(self.left)
        self.layout.addLayout(self.right)

        #   Loaded Image
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(330)
        self.scroll.setMinimumWidth(330)
        self.left.addWidget(self.scroll)

        #       Internal Image
        self.label = QLabel()
        self.pixmap = QPixmap()
        self.label.setPixmap(self.pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.show()
        self.scroll.setWidget(self.label)

        #   File Loader
        self.fileloaders = QHBoxLayout()
        self.left.addLayout(self.fileloaders)

        #       Path
        self.pathfinder = QLineEdit()
        self.fileloaders.addWidget(self.pathfinder)

        #       Button
        self.pathbutton = QPushButton("Load")
        self.filepicker = QFileDialog()
        self.pathbutton.clicked.connect(lambda: self.pressLoad())
        self.fileloaders.addWidget(self.pathbutton)

        #   Options
        self.optionsLayout = QVBoxLayout()
        self.left.addLayout(self.optionsLayout)

        #       Method
        self.methodOption = QComboBox()
        self.methodOption.addItem("Metode Norm (numpy)")
        self.methodOption.addItem("Metode Cosine (numpy)")
        self.methodOption.addItem("Metode Norm")
        self.methodOption.addItem("Metode Cosine")
        self.optionsLayout.addWidget(self.methodOption)

        #       Show Keypoints
        self.showKeypointsLayout = QHBoxLayout()
        self.showKeypointsLabel = QLabel('Tampilkan Keypoint')
        self.showKeypointsLayout.addWidget(self.showKeypointsLabel)
        self.showKeypoints = QCheckBox()
        self.showKeypointsLayout.addWidget(self.showKeypoints)
        self.showKeypointsLayout.setAlignment(Qt.AlignRight)
        self.optionsLayout.addLayout(self.showKeypointsLayout)

        #       Reverse
        self.reverseLabel = QLabel('Balikkan Urutan')
        self.showKeypointsLayout.addWidget(self.reverseLabel)
        self.reverse = QCheckBox()
        self.showKeypointsLayout.addWidget(self.reverse)

        #   List
        self.listLayout = QVBoxLayout()
        self.right.addLayout(self.listLayout)
        self.right.setAlignment(self.listLayout, Qt.AlignTop)

        #           Label
        self.listLabel = QLabel("  Peringkat Gambar Termirip")
        self.listLayout.addWidget(self.listLabel)

        #           List
        self.resultList = QListWidget()
        self.resultList.horizontalScrollBar().setDisabled(True)
        self.resultList.setFixedWidth(355)
        self.resultList.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.resultList.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.listLayout.addWidget(self.resultList)

        #   Controller
        self.controllerLayout = QHBoxLayout()
        self.right.addLayout(self.controllerLayout)

        #       Text
        self.controllerLabel = QLabel('Banyak data yang ingin ditampilkan:')
        self.controllerLayout.addWidget(self.controllerLabel)

        #       Spinner
        self.controllerSpinner = QSpinBox()
        self.controllerSpinner.setRange(1,50)
        self.controllerSpinner.setSingleStep(1)
        self.controllerSpinner.setValue(7)
        self.controllerSpinner.valueChanged.connect(lambda v:self.updateList())
        self.controllerLayout.addWidget(self.controllerSpinner)

        #   Button
        self.start = QPushButton("Cari Gambar Termirip")
        self.start.clicked.connect(lambda:self.compare())
        self.left.addWidget(self.start)
        
        self.show()

# peluncuran program utama ketika kode dijalankan dengan interpreter
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())