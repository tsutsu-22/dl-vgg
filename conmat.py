from keras.models import load_model
import numpy as np
#from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import img_to_array, load_img
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PIL import Image

def predict(jpg_name, model_file_name):

    # img = '識別したい画像ファイル名'
    # model_file_name='重みモデルのファイル名'

    model=load_model(model_file_name)

    img_path = (jpg_name)
    img = img_to_array(load_img(img_path, target_size=(224,224)))
    
    ####

    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]

    label=['f','m']
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]
    print('name:',pred_label)
    print('score:',score)
    
    return pred_label

def conmat(case_n,cv_n,sex):
    print("conmat")

    model_name='case_'+case_n+'/cv_'+cv_n+'/weight/'+cv_n+'.h5'
    dir='case_'+case_n+'/cv_'+cv_n+'/validation/'+sex
    files = glob.glob(dir+"/*")

    true_data=[]
    pred_data=[]

    for file in files:
        true_data.append(sex)
        pred_data.append(predict(file,model_name))

    print(true_data)
    print(pred_data)
    #cm=confusion_matrix(true_data,pred_data)
    #print(cm)

    #return cm
    return true_data,pred_data

def flatten_list(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten_list(el)
        else:
            yield el

def main():
    case="k"
    true=[]
    pred=[]
    for i in range(5):
        tdf,pdf=conmat(case,str(i),"f")
        tdm,pdm=conmat(case,str(i),"m")
        true.append(tdf)
        true.append(tdm)
        pred.append(pdf)
        pred.append(pdm)

    print(true,pred)

    result=classification_report(list(flatten_list(true)), list(flatten_list(pred)))
    print(result)
    dir='case_'+case+'/confusion_matrix'
    open(dir+".txt","w").write(result)


if __name__ == "__main__":
    main()
    