
from keras.models import load_model
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
import glob
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt


def predict(jpg_name, model_file_name):

    # img = '識別したい画像ファイル名'
    # model_file_name='重みモデルのファイル名'

    model=load_model(model_file_name)

    img_path = (jpg_name)
    img = img_to_array(load_img(img_path, target_size=(224,224)))
    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]

    pred = model.predict(img_nad, batch_size=1, verbose=0)
    
    return pred

def conmat(case_n,cv_n,sex):

    model_name='case_'+case_n+'/cv_'+cv_n+'/weight/'+cv_n+'.h5'
    dir='case_'+case_n+'/cv_'+cv_n+'/validation/'+sex
    files = glob.glob(dir+"/*")

    if sex=="f":
        k=0
    else:
        k=1


    true_data=[]
    label=[]

    for file in files:
        true_data.append(sex)
        label.append(predict(file,model_name)[0][k])

    # print(true_data)
    # print(label)

    return true_data,label

def main():
    case="k"

    for i in range(1):
        f_d,f_l=conmat(case,str(i),"f")
        m_d,m_l=conmat(case,str(i),"m")

        # print(f_d,f_l)
        # print(m_d,m_l)

        y_true=f_d
        y_true.extend(m_d)
        y_score=f_l
        y_score.extend(m_l)

    for i in range(1,5):
        f_d,f_l=conmat(case,str(i),"f")
        m_d,m_l=conmat(case,str(i),"m")

        # print(f_d,f_l)
        # print(m_d,m_l)

        y_true.extend(f_d)
        y_true.extend(m_d)
        y_score.extend(f_l)
        y_score.extend(m_l)
    
    # print(y_true)
    # print(y_score)

    for i in range(len(y_true)):
        if y_true[i]=="f":
            y_true[i]=int(0)
        else:
            y_true[i]=int(1)

    print(y_true)
    print(y_score)
    
    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_score,drop_intermediate=False)

    y_true=np.array(y_true)
    y_score=np.array(y_score)

    print(y_true, y_score)
    print(len(y_true), len(y_score))

    

    plt.plot(fpr_all, tpr_all, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('case_'+case+'/sklearn_roc_curve_all.png')
    
    scr=roc_auc_score(y_true, y_score)

    print(scr)

    open('case_'+case+"/auc"+".txt","w").write("auc="+str(scr))


if __name__ == "__main__":
    main()