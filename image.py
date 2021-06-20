import os
import cv2 as cv
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.models import Model,model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Convolution2D
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from skimage.transform import AffineTransform, rotate, warp, warp
class Image():
    def __init__(self):
        self.model = None
        self.model_machine = None
        self.FC_layer_model = None
        self.tfsaved=0
        self.saved=0
        self.classNamesT=['cloudy','rain','shine','sunrise']
    """
    Gerekli değişkenlerin ve kütüphanelerin importu. classNamesT önceden 
    kaydedilen modelin testinde sonuç bulunduktan sonra isimlendirme için.
    """
    def testModel(self,path):
        X=[]
        img = io.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img,(120,120))
        if(np.max(img) > 1):
            img = img/255.
        X.append(img)
        X = np.array(X)  
        X = np.reshape(X,(-1,120,120,3)) 
        json_file = open('./models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./models/model.h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        result = loaded_model.predict(X)            
        etiket=None
        print(result)
        for values in result:
            best_value=0
            for i,value in enumerate(values):
                if (value>best_value):
                    best_value=value
                    etiket=i          
            return (f"En iyi değer: {best_value} Etiket:{self.classNamesT[etiket]} "  )  
    """
    Modelin testi işleminde burası çalışır.(Derin öğrenme)
    Seçilen imagein istenen şekle getirme işlemlerinden sonra önceden kaydedilen model getirilip 
    resme göre tahminleme yaptırılır daha sonra bu tahminlemede dönen değerlerden en büyüğü sonuç 
    olarak isimlendirilip döndürülür
    """
    def testModelTF(self,path):
        X=[]
        img = io.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img,(120,120))
        if(np.max(img) > 1):
            img = img/255.
        X.append(img)
        X = np.array(X)  
        X = np.reshape(X,(-1,120,120,3)) 
        json_file = open('./models/modelTF_json_mnist.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./models/modelTF_mnist.h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        result = loaded_model.predict(X)            
        etiket=None
        for values in result:
            best_value=0
            for i,value in enumerate(values):
                if (value>best_value):
                    best_value=value
                    etiket=i          
            return (f"En iyi değer: {best_value} Etiket:{self.classNamesT[etiket]} "  )  
    """
    Modelin testi işleminde burası çalışır. (Makine öğrenmesi)
    Seçilen imagein istenen şekle getirme işlemlerinden sonra önceden kaydedilen model getirilip 
    resme göre tahminleme yaptırılır daha sonra bu tahminlemede dönen değerlerden en büyüğü sonuç 
    olarak isimlendirilip döndürülür
    """
    def getLabel(self,path):
        label = []
        images = os.listdir(path)
        for img in images:
            new_label = ''
            for i in img:
                if i.isnumeric():
                    continue
                else:
                    new_label +=i
            label.append(new_label.split('.')[0])
        return label
    """
    Dosya yolundaki tüm imagelerin türlerini numaralardan ayrıştırır ve dizide döndürür.(Örneğin winter122 ise winter)
    """
    def get_X_Y(self,path):
        images = os.listdir(path)
        X = []
        Y = self.getLabel(path)
        self.classNames = []
        state= 0
        for i in Y:
            for index,j in enumerate(self.classNames):
                if(i==self.classNames[index]):
                    state = 1
            if(state!=1):
                self.classNames.append(i)
            state=0
        for i in images:
            img = io.imread(path+'/'+str(i))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img,(120,120))
            if(np.max(img) > 1):
                img = img/255.
            X.append(img)
        X = np.array(X)  
        X = np.reshape(X,(-1,120,120,3)) 
        lb = LabelEncoder() 
        Y = lb.fit_transform(Y)
        Xs, Ys = shuffle(X, Y)
        return Xs,Ys,self.classNames
    """
    ilk olarak kendisine verilen pathdeki görüntüleri okur daha sonra getlabel metodunu çağırır bu metodla sınıflar elde edilir.
    cvt color metodu ile görüntünün renk uzayı rgb renk uzayına dönüştürülür. imagein boyutu resize yapılır ve normalization yapılır.
    getlabel ile bulunan sınıflar sayısal verilere dönüştürülür.
    """
    def augmentationImage(self,path):
        transform = AffineTransform(translation=(-200,0))
        images = os.listdir(path)
        for i in images:
            for j in i:
                new_label = ''
                if i.isnumeric():
                    new_label +=j
                    print(j)
                else:
                    new_label +=i
            new_label=new_label.split('.')[0]
            img = io.imread(path+'/'+str(i))   
            hflipped_image= np.fliplr(img)
            vflipped_image= np.flipud(img)
            r_image1 = rotate(img, angle=-45)  
            r_image = rotate(img, angle=45)
            warp_image = warp(img,transform, mode="wrap")
            io.imsave(path+'/'+str(new_label)+str(100)+'.jpg',hflipped_image) 
            io.imsave(path+'/'+str(new_label)+str(101)+'.jpg',vflipped_image)   
            io.imsave(path+'/'+str(new_label)+str(102)+'.jpg',r_image1)   
            io.imsave(path+'/'+str(new_label)+str(103)+'.jpg',r_image)  
            io.imsave(path+'/'+str(new_label)+str(104)+'.jpg',warp_image)  
    """
    Veri arttırma için bu kısım çalışır seçilen klasördeki için farklı 5 resim oluşturur ek olarak.
    """
    def split(self,types,yuzde,X,Y):
        x_train, x_test, y_train, y_test = None,None,None,None
        x_train_k, x_test_k, y_train_k, y_test_k = [],[],[],[]
        index_uzayi = []
        if types == 'Hold Out': 
            x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = yuzde*0.01, random_state = 2)
            return x_train, x_test, y_train, y_test, index_uzayi
        elif types == 'K-Fold':
            kf = KFold(n_splits=yuzde)
            for train_index, test_index in kf.split(X):
                x_train_k, x_test_k, y_train_k, y_test_k = [],[],[],[]
                for i in train_index:
                    x_train_k.append(X[i])
                    y_train_k.append(Y[i])
                for i in test_index:
                    x_test_k.append(X[i])
                    y_test_k.append(Y[i])
                index_uzayi.append([train_index, test_index])
            return np.array(x_train_k), np.array(x_test_k), np.array(y_train_k), np.array(y_test_k), index_uzayi
    """
    Verisetinin eğitim ve test olarak ayrılması.
    """  
    def vgg16(self,nb_classes):
        vgg = VGG16(weights='imagenet', input_shape=(120,120,3), include_top=False)
        model = Sequential()
        for layer in vgg.layers:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = False
        model.add(Flatten())
        model.add(Dense(4096,activation='relu',kernel_constraint=maxnorm(3), name = 'dense1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation='relu',kernel_constraint=maxnorm(3),name = 'dense2'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes,activation='softmax',name = 'outPut'))
        model.summary()
        return model
    """
    vgg16 algoritması bu kısım ile transfer learning yapılır ve derin öğrenme uygulanır.
    """
    def AlexNet(self,nb_classes):
        shape = (120,120,3)
        X_input = Input(shape)
        X = Conv2D(96,(11,11),strides = 4,name="conv0")(X_input)
        X = BatchNormalization(axis = 3 , name = "bn0")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
        X = Conv2D(256,(5,5),padding = 'same' , name = 'conv1')(X)
        X = BatchNormalization(axis = 3 ,name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
        X = Conv2D(384, (3,3) , padding = 'same' , name='conv2')(X)
        X = BatchNormalization(axis = 3, name = 'bn2')(X)
        X = Activation('relu')(X)
        X = Conv2D(384, (3,3) , padding = 'same' , name='conv3')(X)
        X = BatchNormalization(axis = 3, name = 'bn3')(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3,3) , padding = 'same' , name='conv4')(X)
        X = BatchNormalization(axis = 3, name = 'bn4')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
        X = Flatten()(X)
        X = Dense(4096, activation = 'relu', name = "dense1")(X)
        X = Dense(4096, activation = 'relu', name = 'fc1')(X) 
        X = Dense(nb_classes,activation='softmax',name = 'fc2')(X)
        model = Model(inputs = X_input, outputs = X, name='AlexNet')
        model.summary()
        return model
    """
    AlexNet algoritması bu kısım ile transfer learning yapılır ve derin öğrenme uygulanır.
    """
    def fit(self,model_name,X,Y):
        if model_name == 'VGG16':
            self.model = self.vgg16(Y.shape[1])
        elif model_name == 'AlexNet':
            self.model = self.AlexNet(Y.shape[1])
        lrate = 0.001
        decay = lrate/100
        sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = self.model.fit(X,Y,validation_split=0.5, epochs=3, batch_size=16)
        return history
    """
    Seçili olana göre transfer learning ve derin öğrenme yapılır daha sonra optimizasyon yapılır ve fit ile eğitilir.
    """    
    def predict(self,model_name,X):
        result = self.model.predict(X)
        return result
    """
    Tahminleme cnn tagı için.
    """
    def transfer_learning_fit(self,model_CNN,model_Machine,X,Y,y):
        history = self.fit(model_CNN,X,Y)
        layer_name = 'dense1'
        self.FC_layer_model = Model(inputs=self.model.input,outputs=self.model.get_layer(layer_name).output)
        features=np.zeros(shape=(X.shape[0],4096))
        for i in range(len(X)):
            img = np.expand_dims(X[i], axis=0)
            FC_output = self.FC_layer_model.predict(img)
            features[i]=FC_output
        feature_col=[]   
        for i in range(4096):
            feature_col.append("f_"+str(i))
            i+=1
        train_features = pd.DataFrame(data=features,columns=feature_col)
        
        print('machine learning')
        if model_Machine == 'KNN':
            self.model_machine =  OneVsRestClassifier(KNeighborsClassifier(5,weights='distance'))
            self.model_machine.fit(train_features,y)
        elif model_Machine == 'Random Forest':
            self.model_machine =  OneVsRestClassifier(RandomForestClassifier(random_state=0))
            self.model_machine.fit(train_features,y)
        return history
    """
    Seçili olana göre transfer learning yapılır ancak derin öğrenme uygulanmaz daha sonra optimizasyon yapılır, seçili sınıflandırma ile 
    sınıflandırma yapılır ve fit ile eğitilir.
    """    
    def transfer_learning_predict(self,X):
        features=np.zeros(shape=(X.shape[0],4096))
        for i in range(len(X)):
            img = np.expand_dims(X[i], axis=0)
            FC_output = self.FC_layer_model.predict(img)
            features[i]=FC_output
        feature_col=[]   
        for i in range(4096):
            feature_col.append("f_"+str(i))
            i+=1
        test_features = pd.DataFrame(data=features,columns=feature_col)
        predicts = self.model_machine.predict(test_features)
        pred_prob = self.model_machine.predict_proba(test_features)
        return pred_prob,predicts
    """
    Tahminleme işlemi yapılır makine öğrenmesi için.
    """
    def save_model_tf(self):
        sModel=self.model_machine
        model_json = sModel.to_json()
        with open("./models/modelTF_mnist.json", "w") as json_file:
            json_file.write(model_json)
        sModel.save_weights("./models/modelTF_json_mnist.h5")
        return("Model(Machine Learning) Kaydedildi.")
    """
    Makine öğrenmesi kısmı için model kayıt işlemi.
    """
    def save_model(self):
        sModel=self.model
        model_json = sModel.to_json()
        with open("./models/model.json", "w") as json_file:
            json_file.write(model_json)
        sModel.save_weights("./models/model.h5")
        return("Model Kaydedildi.")
    """
    Derin öğrenmeli kısım için model kayıt işlemi.
    """      
    def clahe(self,path):
    
        if os.path.exists(str(os.getcwd())+"/clahe") == False:
            os.mkdir("clahe")
        images = os.listdir(path)
        claheIMG = None
        for i in images:
            img = io.imread(path+'/'+str(i))
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            R, G, B = cv.split(img)
            clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
            claheIMG_R = clahe.apply(R)
            claheIMG_G = clahe.apply(G)
            claheIMG_B = clahe.apply(B)
            claheIMG = cv.merge((claheIMG_R,claheIMG_G,claheIMG_B))
            cv.imwrite('./clahe/'+str(i),claheIMG) 
    """
    İlk olarak clahe dosyası yoksa dosyayı oluşturur dönüştürülen imageleri kaydetmek için
    Daha sonra kendisine verilen path deki imageleri tek tek okur.
    cvt color metodu ile görüntünün renk uzayı rgb renk uzayına dönüştürülür.
    Clahe sınıfına kontrast sınırlama eşiğini 2 ve ızgara boyutunu 8,8 vererek işaretçi oluşturulur
    rgb deki tüm renk uzayları için clahe metodunu uygulanır
    uygulanmış tüm değişiklikleri birleştirilir ve görüntüyü kaydedilir.
    """
    def HE(self,path):
        if os.path.exists(str(os.getcwd())+"/HE") == False:
            os.mkdir("HE")
        images = os.listdir(path)
        for i in images:
            img = io.imread(path+'/'+str(i))
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img = self.histogram_equalization(img)
            cv.imwrite('./HE/'+str(i), img) 
    """
    İlk olarak he dosyası yoksa dosyayı oluşturur dönüştürülen imageleri kaydetmek için
    Daha sonra kendisine verilen path deki imageleri tek tek okur. 
    RGB ye dönüşütüp histogram_equalization metodunu çalıştırır ve imageleri he ye dönüştürmüş olur.Dönüştürülen resmi kaydeder.
    """
    def histogram_equalization(self,img_in):
        b,g,r = cv.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
        cdf_b = np.cumsum(h_b)  
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)
        cdf_m_b = np.ma.masked_equal(cdf_b,0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
        cdf_m_g = np.ma.masked_equal(cdf_g,0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
        cdf_m_r = np.ma.masked_equal(cdf_r,0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]
        img_out = cv.merge((img_b, img_g, img_r))
        equ_b = cv.equalizeHist(b)
        equ_g = cv.equalizeHist(g)
        equ_r = cv.equalizeHist(r)
        equ = cv.merge((equ_b, equ_g, equ_r))
        return img_out
    """
    histogram eşitleme işlemi yapılır tüm renk uzayı için.
    """
    def graphROC(self, y_test, pred_prob,classNames):
        n_class=4
        fpr = {}
        tpr = {}
        thresh ={}
        colors = ['orange','green','blue']
        for i in range(n_class):   
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        k=2
        for j in range(n_class):   
            if(k==3):
                k=0
            plt.plot(fpr[j], tpr[j], linestyle='--',color=str(colors[k]), label='ROC curve of class {0}'''.format(j))
            k=k+1
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC curve')
        plt.legend(loc='best')
        plt.savefig('./graphic/graphROC.png',dpi=100);
    """
    Rok öğreisi oluşturulur ve kaydedilir.
    """
    def graphAccuracyLoss(self,accuracy,loss):
        fig,axes = plt.subplots(nrows=1, ncols=1, figsize=(7,2))
        axes.plot(accuracy)
        axes.plot(loss)
        axes.set_title('Deep Learning')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Accuracy and Loss')
        axes.legend(['Accuracy', 'Loss'],loc='upper right')
        fig.savefig('./graphic/graphAccuracyLoss.png')
    """
    AccuracyLoss grafiği oluşturulur ve kaydedilir.
    """
    def graphConfusionMatrix(self,y_true, y_test):
            return confusion_matrix(y_test,y_true)  
    """
    confusion matrix oluşturulur.
    """