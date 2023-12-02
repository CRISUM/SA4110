import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# transfer learning
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# 创建一个 ImageDataGenerator 实例
datagen = ImageDataGenerator(
    validation_split=0.2,
    horizontal_flip=True,  # 开启水平翻转
    zoom_range=[0.8, 1.2],   # 开启随机缩放
    fill_mode='nearest'
)

def get_onehot(label):
    
    if label == 'apple': _1hot = [1, 0, 0, 0]
    elif label == 'banana': _1hot = [0, 1, 0, 0]
    elif label == 'orange': _1hot = [0, 0, 1, 0]
    else: _1hot = [0, 0, 0, 1]
    # converting from python list to numpy array
    return np.array([np.asarray(_1hot)])

def getPicPath(ori_path,jud):
    file_list = os.listdir(ori_path+jud+"/")
    ori_xml_list=[]
    for file in file_list:
        if(os.path.splitext(file)[1]=='.jpg'):
            #print('pic file found.')
            #print(file)
            ori_xml_list.append(file)
    return ori_xml_list

def process_image(image_path, size=(120,120)):
    with Image.open(image_path) as img:
        img = img.resize(size)
        # 转换为 RGB 格式（如果需要）
        img_rgb = img.convert('RGB')
        # 将图像数据转换为 numpy 数组
        img_array = np.array([np.asarray(img_rgb)])
        return img_array

def get_label_from_filename(filename):
    # 提取文件名前缀作为标签，具体实现可能根据你的文件名格式而有所不同
    label = os.path.basename(filename).split('_')[0]
    return label

def getPicData(ori_path,jud):
    file_path = getPicPath(ori_path,jud)
    for file in file_path:
        try: 
            x_data = np.concatenate((x_data,process_image(ori_path+jud+"/"+file)),)
        except: 
            x_data = process_image(ori_path+jud+"/"+file)
        try: 
            y_data = np.concatenate((y_data,get_onehot(get_label_from_filename(file))))
        except: 
            y_data = get_onehot(get_label_from_filename(file))
    return x_data,y_data

def create_model():

    model = tf.keras.models.Sequential([
        # 卷积层
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(120, 120, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # 更多的卷积层
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 防止过拟合
        tf.keras.layers.Dropout(0.1),

        # 全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),

        # 输出层
        tf.keras.layers.Dense(4, activation='softmax')
    ])    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model
    
# transfer learning
def create_model2():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(120, 120, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 冻结 MobileNet 的层
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def plotSeparate(hist):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(hist.history['loss'], label='loss')
    ax[0].plot(hist.history['val_loss'], label='val_loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')
    ax[0].legend()

    ax[1].plot(hist.history['accuracy'], label='accuracy')
    ax[1].plot(hist.history['val_accuracy'],  label='val_loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend()

    plt.show()    
   

def plotCombine(hist):
    plt.subplots(figsize=(6, 5))

    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['accuracy'], label='accuracy') 
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.plot(hist.history['val_accuracy'], label='val_accuracy')
    
    plt.xlabel('Epochs')

    plt.legend()
    plt.show()   

def main():
    model = create_model()
    
    tst = os.path.basename(__file__)
    print(__file__[:-len(tst)])
    path = __file__[:-len(tst)]
    
    (x_train,y_train) = getPicData(path,'train_ori')
    (x_test,y_test) = getPicData(path,'test')
    
    print(x_train.shape,x_train.size)
    print(y_train.shape,y_train.size)
    
    #x_train = np.reshape(x_train, (x_train.shape[0], 100, 100, 1))
    #x_test = np.reshape(x_test, (x_test.shape[0], 100, 100, 1))
    
    #y_train - tf.keras.utils.to_categorical(y_train, 3)
    #y_test - tf.keras.utils.to_categorical(y_test, 3)
    
    #print(x_train,y_train)
    #print(x_test,y_test)
    
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,stratify=y_train)

    train_generator = datagen.flow(x_train/55.1, y_train, batch_size=10, subset="training", shuffle=True)
    validation_generator = datagen.flow(x_train/55.1, y_train, batch_size=10, subset="validation", shuffle=True)

    steps_per_epoch = train_generator.n//train_generator.batch_size    

    hist = model.fit(x=train_generator, validation_data=validation_generator, steps_per_epoch=steps_per_epoch, epochs=10)

    # showing how we can save our trained model
    model.save(path+'model')
    
    # showing how we can load our trained model
    model = tf.keras.models.load_model(path+'model')

    # test how well our model performs against data
    # that it has not seen before
    model.evaluate(x=x_test/55.1, y=y_test)

    # 假设你有一个包含图像路径的列表
    plotSeparate(hist)
    plotCombine(hist)
   

if __name__ == "__main__":
    main()