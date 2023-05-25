import glob
import shutil
import random

sex="f"
case_n="k"
files = glob.glob("case_"+case_n+"/ori/"+sex+"/*jpg")

num=int(len(files))

#シャッフル
random.shuffle(files)
i=0
for file in files:
    if i < num/5*4:
        shutil.copy2(file, 'case_'+case_n+'/cv_0/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_0/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*3 or i >= num/5*4:
        shutil.copy2(file, 'case_'+case_n+'/cv_1/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_1/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*2 or i >= num/5*3:
        shutil.copy2(file, 'case_'+case_n+'/cv_2/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_2/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*1 or i >= num/5*2:
        shutil.copy2(file, 'case_'+case_n+'/cv_3/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_3/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*0 or i >= num/5*1:
        shutil.copy2(file, 'case_'+case_n+'/cv_4/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_4/validation/'+sex+'/')
    i=i+1


sex="m"
case_n="k"
files = glob.glob("case_"+case_n+"/ori/"+sex+"/*jpg")

num=int(len(files))

#シャッフル
random.shuffle(files)
i=0
for file in files:
    if i < num/5*4:
        shutil.copy2(file, 'case_'+case_n+'/cv_0/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_0/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*3 or i >= num/5*4:
        shutil.copy2(file, 'case_'+case_n+'/cv_1/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_1/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*2 or i >= num/5*3:
        shutil.copy2(file, 'case_'+case_n+'/cv_2/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_2/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*1 or i >= num/5*2:
        shutil.copy2(file, 'case_'+case_n+'/cv_3/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_3/validation/'+sex+'/')
    i=i+1

i=0
for file in files:
    if i < num/5*0 or i >= num/5*1:
        shutil.copy2(file, 'case_'+case_n+'/cv_4/train/'+sex+'/')
    else:
        shutil.copy2(file, 'case_'+case_n+'/cv_4/validation/'+sex+'/')
    i=i+1