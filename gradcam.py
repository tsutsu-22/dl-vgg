from __future__ import print_function
import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras import backend as K 
import os
import numpy as np  
import glob  
import pandas as pd
import cv2
import matplotlib.pyplot as plt

K.set_learning_phase(1) #set learning phase

def Grad_Cam(input_model, pic_array, layer_name):

  # 前処理
  pic = np.expand_dims(pic_array, axis=0)
  pic = pic.astype('float32')
  preprocessed_input = pic / 255.0

  # 予測クラスの算出
  predictions = input_model.predict(preprocessed_input)
  print(predictions)
  print(predictions[0].argmax())
  class_idx = np.argmax(predictions[0])
  class_output = input_model.output[:, class_idx]

  #  勾配を取得
  conv_output = input_model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
  grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
  gradient_function = K.function([input_model.input], [conv_output, grads])  # input_model.inputを入力すると、conv_outputとgradsを出力する関数

  output, grads_val = gradient_function([preprocessed_input])
  output, grads_val = output[0], grads_val[0]

  # 重みを平均化して、レイヤーのアウトプットに乗じる
  weights = np.mean(grads_val, axis=(0, 1))
  cam = np.dot(output, weights)

  # 画像化してヒートマップにして合成
  cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR) 
  cam = np.maximum(cam, 0) 
  cam = cam / cam.max()

  jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
  jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
  jetcam = (np.float32(jetcam) + pic / 2)   # もとの画像に合成
  return jetcam

def all(dir,dr,out_path,model):
  files = glob.glob(dir+"/*")
  for file in files:
    
    fig = plt.figure()
    X=1
    Y=2

    print(file)

    pic_array = img_to_array(load_img(file, target_size=(224, 224)))
    #pic = pic_array.reshape((1,) + pic_array.shape)
    fig.add_subplot(X, Y, 1)
    plt.imshow(pic_array/np.amax(pic_array))

    array_to_img(pic_array)
    picture = Grad_Cam(model, pic_array, 'block5_conv3')
    picture = picture[0,:,:,]
    fig.add_subplot(X, Y, 2)
    plt.imshow(picture/np.amax(picture))
    #plt.show()
    f=file.replace(dir,'')
    
    output_path = dr+out_path
    if os.path.isdir(output_path) == False:
      os.mkdir(output_path)
    fig.savefig(output_path+"/"+f)



def main():
  nums=['0','1','2','3','4'] #cv_n
  #train,validationすべてにgradcam
  case_n='k'
  for nm in nums:
    dr='case_'+case_n+'/cv_'+nm+'/'
    #モデルのロード
    model=keras.models.load_model(dr+'weight/'+nm+'.h5')

    dir=dr+'train/f'
    all(dir,dr,'gradcam_t',model)

    dir=dr+'validation/f'
    all(dir,dr,'gradcam_v',model)

    dir=dr+'train/m'
    all(dir,dr,'gradcam_t',model)

    dir=dr+'validation/m'
    all(dir,dr,'gradcam_v',model)

if __name__ == "__main__":
    main()