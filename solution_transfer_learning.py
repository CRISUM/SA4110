import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import string
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# transfer learning
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model



def encode_onehot(type):
    # 4 classes (digit 0 to 3)
    y_onehot = [0] * 4
    
    if type == "apple":
        y_onehot[0] = 1
    if type == "banana":
        y_onehot[1] = 1
    if type == "orange":
        y_onehot[2] = 1 
    if type == "mixed":
        y_onehot[3] = 1       
    
    return np.array([y_onehot])

# 把所有的图片叠在一起 把所有的标签转成one hot叠在一起 
def read_img_data(path):
    x = np.empty((0, 128, 128, 3))  # 初始化x
    y = np.empty((0,))            # 初始化y
    for file in os.listdir(path):
        if file[0] == '.':
            continue
        
        # 处理one hot标签
        if file.startswith("apple"):
           y_onehot = encode_onehot("apple")
        if file.startswith("banana"):
           y_onehot = encode_onehot("banana")
        if file.startswith("orange"):
           y_onehot = encode_onehot("orange")
        if file.startswith("mixed"):
           y_onehot = encode_onehot("mixed")           
        try:
            y = np.concatenate((y, y_onehot))
        except:
            y = y_onehot

        # 处理特征部分
        img = Image.open("{}/{}".format(path, file)).convert('RGB')
        img = img.resize((128, 128))
        data = np.array([np.asarray(img)])
        try:
            x = np.concatenate((x,data))
        except:
            x = data
        
    return x,y

def plot(hist):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(hist.history['loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')

    ax[1].plot(hist.history['accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')

    ax[0].plot(hist.history['val_loss'])
    ax[1].plot(hist.history['val_accuracy'])

    plt.legend()
    plt.show()    

'''
def create_model():
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))
    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dropout(0.2))

    #model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    #model.add(tf.keras.layers.Dense(units=256, activation='sigmoid'))

    # 防止过拟合，简化模型
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))

    model.add(tf.keras.layers.Dense(units=4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 收敛速度太慢
    return model
'''

# transfer learning
def create_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 冻结 MobileNet 的层
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    x_train, y_train = read_img_data('train/')
    x_test, y_test = read_img_data("test/")

    datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

    # print(x_train.shape)

    model = create_model()
    print(model.summary())

    # model.fit(x=x_train/255, y=y_train, epochs=15)
    hist = model.fit(datagen.flow(x_train/255, y_train, batch_size=32), epochs=50, validation_data = (x_test,y_test))
    # plot(hist)

    
    model.save('mnist_saved_model/')
    model = tf.keras.models.load_model('mnist_saved_model')
    
    model.evaluate(x=x_test/255, y=y_test)

if __name__ == '__main__':
    main()
