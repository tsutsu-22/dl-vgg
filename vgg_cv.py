from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

import tensorflow as tf
import numpy as np
import random
import os

#シード固定
def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(0)

#############パラメータ入力###################
case="k"
k_fold=5
n_categories=2
batch_size=2
lay=11
epoch=5
ir=0.001
momentum=0.9
loss_f='categorical_crossentropy'
############################################


def Vgg(train_dir,validation_dir,file_name):
    print("vgg")
    base_model=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

    #add new layers instead of FC networks
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    prediction=Dense(n_categories,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=prediction)

    #fix weights before VGG16 14layers
    for layer in base_model.layers[:lay]:
        layer.trainable=False

    model.compile(optimizer=SGD(lr=ir,momentum=momentum),loss=loss_f,metrics=['accuracy'])
    model.summary()

    #save model
    json_string=model.to_json()
    open(file_name+'.json','w').write(json_string)

    #学習
    train_datagen=ImageDataGenerator(rescale=1.0/255,)
    validation_datagen=ImageDataGenerator(rescale=1.0/255)
    train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # model.compile(optimizer=SGD(lr=ir,momentum=0.9),
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])
    
    model.fit_generator(train_generator,
                        epochs=epoch,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=[CSVLogger(file_name+'.csv')])

    #save weights
    model.save(file_name+'.h5')

def cv(case_n,cv_n):
    print("cv")
    dir="case_"+case_n+"/cv_"+cv_n+"/"
    train=dir+"train"
    val=dir+"validation"
    save_file=dir+"weight/"+cv_n

    Vgg(train,val,save_file)

def main():
    for i in range(k_fold):
        cv(case,str(i))

if __name__ == "__main__":
    main()


