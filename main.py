import sys
import os
import numpy as np
import pandas as pd
import time
from PyQt5 import QtWidgets, QtGui
from Form import Ui_MainWindow
from image import Image
from keras.utils import to_categorical
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.df = pd.DataFrame()
        self.y_tr_train, self.y_tr_test, self.x_train, self.x_test, self.y_train, self.y_test, self.X, self.Y = None, None, None, None, None, None, None, None
        self.cls = None
        self.path = None
        self.n_class = 0
        self.classNames= None
        if os.path.exists(str(os.getcwd())+"/graphic") == False:
            os.mkdir("graphic")
        self.ui.termenal.addItem('Windows PowerShell')
        self.ui.termenal.addItem('Copyright (C) Microsoft Corporation. All rights reserved.')
        self.ui.termenal.addItem(' ')
        self.ui.btn_open_file.clicked.connect(self.openFile)
        self.ui.btn_uygula_split.clicked.connect(self.split)
        self.ui.btn_fit_CNN.clicked.connect(self.fit)
        self.ui.btn_predict_CNN.clicked.connect(self.predict)
        self.ui.btn_CLAHE.clicked.connect(self.CLAHE)
        self.ui.btn_HE.clicked.connect(self.HE)
        self.ui.btn_fit_CNN_2.clicked.connect(self.fit_tf)
        self.ui.btn_predict_CNN_2.clicked.connect(self.predict_tf)
        self.ui.btn_savemodelCNN.clicked.connect(self.save_model_cnn)
        self.ui.btn_savemodelM.clicked.connect(self.save_model_m)
        self.ui.btn_resimSec.clicked.connect(self.resimSec)
        self.ui.btn_resimSecTF.clicked.connect(self.resimSecTF)
        self.ui.btn_veriarttir.clicked.connect(self.augmentation)
    """
    gerekli kütüphanlerin importu, 
    değişkenlerin oluşturulması, oluşturulan graafikleri kaydetmek için klasör oluşturma ve butonların fonksiyonlarının tanımlanması kısmı.
    """
    def augmentation(self):
        self.ui.termenal.addItem('Bekle......')
        fileName = QtWidgets.QFileDialog.getExistingDirectory(
        self, "Veri arttırımı yapılacak klasörü seç", str(os.getcwd()))
        self.ui.txt_file_name.setText(fileName)
        self.path = fileName
        self.cls = Image()
        self.cls.augmentationImage(self.path)
        self.ui.termenal.addItem('Veriler Arttırıldı......')
        self.X, self.Y,self.classNames = self.cls.get_X_Y(fileName)
        self.ui.termenal.addItem('X ve Y oluşturuldu......')
    """
    Veri arttır butonuna tıklanırsa burası çalışır.
    Veri arttırılacak klasör seçilir image.py deki augmentationImage metodu ile görüntüler çağaltılır daha sonra get_X_Y ile x ve y elde edilir.
    """
    def openFile(self):
        self.ui.termenal.addItem('Bekle......')
        fileName = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Veri Setini Seç", str(os.getcwd()))
        self.ui.txt_file_name.setText(fileName)
        self.path = fileName
        self.cls = Image()
        self.X, self.Y,self.classNames = self.cls.get_X_Y(fileName)
        self.ui.termenal.addItem('X ve Y oluşturuldu......')
    """
    İşlemlerin yapılacağı dosyayı seçme işlemi seçilen dosyadaki verilein get_X_Y ile x ve y leri elde edilir.
    """
    def resimSec(self):
        self.ui.termenal.addItem('Bekle......')
        fileName = QtWidgets.QFileDialog.getOpenFileName(
            self, "Resmi Seç", str(os.getcwd()))
        self.cls = Image()
        self.ui.termenal.addItem('Resim Seçildi......')
        self.ui.termenal.addItem('Test Ediliyor......')
        sonuc=self.cls.testModel(fileName[0])
        self.ui.label_class.setText(sonuc)
    """
    CNN tagında test işlemi için butona tıklandığında burası çalışır. 
    Başta resim seçilir.
    Seçilen resim testmodel ile test edilir ve ekranda sonuç yazılır.
    """
    def resimSecTF(self):
        self.ui.termenal.addItem('Bekle......')
        fileName = QtWidgets.QFileDialog.getOpenFileName(
            self, "Resmi Seç", str(os.getcwd()))
        self.cls = Image()
        self.ui.termenal.addItem('Resim Seçildi......')
        self.ui.termenal.addItem('Test Ediliyor......')
        sonuc=self.cls.testModelTF(fileName[0])
        self.ui.label_class.setText(sonuc)
    """
    CNN with machine learning tagında test işlemi için butona tıklandığında burası çalışır. 
    Başta resim seçilir.
    Seçilen resim testmodel ile test edilir ve ekranda sonuç yazılır.
    """
    def CLAHE(self):
        self.ui.termenal.addItem('Bekle......')
        self.ui.termenal.addItem('CLAHE oluşturuluyor......')
        self.cls.clahe(self.path)
        self.ui.termenal.addItem('CLAHE oluşturuldu......')
        self.X, self.Y,self.classNames = self.cls.get_X_Y('./clahe')
        self.ui.termenal.addItem('X ve Y oluşturuldu......')
    """
    CLAHE butonuna tıklandığında çalışan metod.
    Ekrana alakalı girdileri yazdıktan sonra image.py deki clahe metodunu çalıştırır bu metodla tüm resimler clahe klasörüne kaydedilir. 
    Daha sonra image.py deki get_X_Y metodu çalıştırılır bu sayede imageler gerekli özelliklerle x olarak ve imagin sınıfı y olarak döner.
    """
    def HE(self):
        self.ui.termenal.addItem('Bekle......')
        self.ui.termenal.addItem('HE oluşturuluyor......')
        self.cls.HE(self.path)
        self.ui.termenal.addItem('HE oluşturuldu......')
        self.X, self.Y,self.classNames, = self.cls.get_X_Y('./HE')
        self.ui.termenal.addItem('X ve Y oluşturuldu......')
    """
    HE butonuna tıklandığında çalışan metod.
    Ekrana alakalı girdileri yazdıktan sonra image.py deki he metodunu çalıştırır bu metodla tüm resimler he klasörüne kaydedilir. 
    Daha sonra image.py deki get_X_Y metodu çalıştırılır bu sayede imageler gerekli özelliklerle x olarak ve imagin sınıfı y olarak döner.
    """
    def split(self):
        self.ui.termenal.addItem('Bekle......')
        item_split = self.ui.cmb_split.currentText()
        yuzde = int(self.ui.yuzde.text())
        cls_n = to_categorical(self.Y)
        self.x_train, self.x_test, self.y_train, self.y_test, self.a = self.cls.split(item_split,yuzde,self.X, self.Y)
        self.y_tr_train = self.y_train
        self.y_tr_test = self.y_test
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.ui.termenal.addItem('x_train, x_test, y_train, y_test oluşturuldu......')
        self.ui.termenal.addItem(f"Sınıf Sayısı : {cls_n.shape[1]}")
        self.ui.termenal.addItem(f"Train için {str(round((self.x_train.shape[0]*100)/self.Y.shape[0]))}% ayrıldı, Test için {str(round((self.x_test.shape[0]*100)/self.Y.shape[0]))}% ayrıldı") 
        self.ui.termenal.addItem(f"Train'da {str(self.x_train.shape[0])} veri var, Test'da {str(self.x_test.shape[0])} veri var") 
    """
    uygulaya basıldığında burası çalışır. 
    Ekranda yazılı olann yüzde değeri, metod ve sınıflar image.pydeki split metoduna verilir bu metodla ilgili olan split 
    metodu(kfold veya holdout) dan dönen veriler kaydedilir daha sonra bunu to_categorical metodu ile onehot encoding yapıp(1 ve 0 lar) dizilere kaydedilir.
    """
    def fit(self):
        self.ui.termenal.addItem('Bekle......')
        self.ui.termenal.addItem('Eğitiliyor......')
        model_name = self.ui.cmb_CNN.currentText()
        history = self.cls.fit(model_name,self.x_train,self.y_train)
        self.cls.graphAccuracyLoss(history.history['accuracy'], history.history['loss'])
        pixmap = QtGui.QPixmap('./graphic/graphAccuracyLoss.png')
        self.ui.label_2.resize(pixmap.width(),pixmap.height())
        self.ui.label_2.setPixmap(pixmap)
        self.ui.termenal.addItem('Eğtim Bitti......')
    """
    cnn kısmında fite basılınca bu kod çalışır.
    ekrandaki seçili olan cnn algoritması ve split metodunda oluşturulan train ve test dizileri image.py deki fit metoduna parametre olarak verilir
    bu metodla elde edilen history image.py deki graphAccuracyLoss metodu ile grafik oluşturup kayededilir daha sonra bu grafik ekranda gösterilir.
    """

    def predict(self):
        self.ui.termenal.addItem('Bekle......')
        self.ui.termenal.addItem('Predict......')
        model_name = self.ui.cmb_CNN.currentText()
        result = 0
        x_test = []
        y_test = []
        y_true = []
        cms = []
        y = to_categorical(self.Y)
        if self.ui.cmb_split.currentText() == 'K-Fold':
            for i in self.a:
                for j, index in enumerate(i):
                    x_test = []
                    y_test = []
                    y_true = []
                    if j == 1:
                        for k in index:
                            x_test.append(self.X[k])
                            y_test.append(y[k])
                            y_true.append(self.Y[k])
                        x_test = np.array(x_test)
                        y_test = np.array(y_test)
                        y_true = np.array(y_true)
                        result = self.cls.predict(model_name,x_test)
                        y_pred = []
                        for i in result:
                            temp = []
                            for j in i:
                                if j < 0.5:
                                    temp.append(0)
                                elif j >= 0.5:
                                    temp.append(1)
                            a = False
                            for v,k in enumerate(temp):
                                if k == 1:
                                    a = True
                                    y_pred.append(v)
                            if a == False:
                                y_pred.append(0)
                        cms.append(self.cls.graphConfusionMatrix(y_true,y_pred))
            cms = np.array(cms)
            cm_n = np.zeros(cms[0].shape)
            for c in cms:
                cm_n = np.add(cm_n,c)
            for i_ in cm_n:
                r = ' '
                for j_ in i_:
                    r +='     '+str(j_)
                self.ui.listWidget.addItem(r)
        else:
            result = self.cls.predict(model_name,self.x_test)
            y_pred = []
            for i in result:
                temp = []
                for j in i:
                    if j < 0.5:
                        temp.append(0)
                    elif j >= 0.5:
                        temp.append(1)
                a = False
                for v,k in enumerate(temp):
                    if k == 1:
                        a = True
                        y_pred.append(v)
                if a == False:
                    y_pred.append(0)
            cm = self.cls.graphConfusionMatrix(self.y_tr_test,y_pred)
            for i_ in cm:
                r = ' '
                for j_ in i_:
                    r +='     '+str(j_)
                self.ui.listWidget.addItem(r)
        self.ui.termenal.addItem('Predict Bitti......')
    """
    cnn kısmında predicte basılınca bu kod çalışır.  Ekranda seçili olan algooritma değişkene atılır sınıflar 
    K-fold için overlop matris holdout için confusion matris elde edilir ve ekranda gösterilir.
    """
    def fit_tf(self):
        self.ui.termenal.addItem('Bekle......')
        self.ui.termenal.addItem('Eğitiliyor......')
        model_CNN = self.ui.cmb_CNN_1.currentText()
        model_Machine = self.ui.cmb_machine.currentText()
        history = self.cls.transfer_learning_fit(model_CNN,model_Machine,self.x_train,self.y_train,self.y_tr_train)
        self.cls.graphAccuracyLoss(history.history['accuracy'], history.history['loss'])
        pixmap = QtGui.QPixmap('./graphic/graphAccuracyLoss.png')
        self.ui.label_3.resize(pixmap.width(),pixmap.height())
        self.ui.label_3.setPixmap(pixmap)
        self.ui.termenal.addItem('Eğtim Bitti......')
    """
    cnn with machine learning kısmında fite basılınca bu kod çalışır.
    ekrandaki seçili olan cnn algoritması, makine öğrenmesi algoritması ve split metodunda oluşturulan train ve test dizileri
    image.py deki transfer_learning_fit metoduna parametre olarak verilir
    bu metodla elde edilen history image.py deki graphAccuracyLoss metodu ile grafik oluşturup kayededilir daha sonra bu grafik ekranda gösterilir.
    """   
    def predict_tf(self):
        self.ui.termenal.addItem('Bekle......')
        self.ui.termenal.addItem('Eğitiliyor......')
        predict = None
        cm_n = []
        cms = []
        x_test = []
        y_test = []
        pred_prob=None
        if self.ui.cmb_split.currentText() == 'K-Fold':
            for i in self.a:
                for j, index in enumerate(i):
                    x_test = []
                    y_test = []
                    if j == 1:
                        for k in index:
                            x_test.append(self.X[k])
                            y_test.append(self.Y[k])
                        x_test = np.array(x_test)
                        y_test = np.array(y_test)
                        predict = self.cls.transfer_learning_predict(x_test)
                        cms.append(self.cls.graphConfusionMatrix(y_test,predict))
            cms = np.array(cms)
            cm_n = np.zeros(cms[0].shape)
            for c in cms:
                cm_n = np.add(cm_n,c)
            for i_ in cm_n:
                r = ' '
                for j_ in i_:
                    r +='     '+str(j_)
                self.ui.listWidget_3.addItem(r)
        else:
            pred_prob,predict = self.cls.transfer_learning_predict(self.x_test)
            cm = self.cls.graphConfusionMatrix(self.y_tr_test,predict)
            for i_ in cm:
                r = ' '
                for j_ in i_:
                    r +='     '+str(j_)
                self.ui.listWidget_3.addItem(r)
        self.cls.graphROC(self.y_tr_test, pred_prob,self.classNames)
        pixmap = QtGui.QPixmap('./graphic/graphROC.png')
        self.ui.lb_roc2.resize(pixmap.width(),pixmap.height())
        self.ui.lb_roc2.setPixmap(pixmap)
        self.ui.termenal.addItem('Eğtim Bitti......')
    """
    cnn kısmında predicte basılınca bu kod çalışır.  Ekranda seçili olan algooritma değişkene atılır sınıflar 
    K-fold için overlop matris holdout için confusion matris elde edilir ve ekranda gösterilir. 
    Roc eğrisi elde eden image.py deki graphROC metodu çalıştırılır ve roc eğrisi ekranda gösterilir.
    """
    def save_model_cnn(self):
        returnS = self.cls.save_model()
        self.ui.termenal.addItem(f"{str(returnS)}")
    """
    Cnn de model kaydede tıklandığında image.py deki modeli kaydeden save_model motodu çalışır.
    """
    def save_model_m(self):
        returnS = self.cls.save_model_tf()
        self.ui.termenal.addItem(f"{str(returnS)}")
    """
    Cnn with machine learning de model kaydede tıklandığında image.py deki modeli kaydeden save_model motodu çalışır.
    """
def app():
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())


app()
