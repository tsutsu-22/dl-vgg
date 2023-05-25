import os

case_name='case_k'

os.makedirs(case_name, exist_ok=True)
os.makedirs(case_name+"/ori/f", exist_ok=True)
os.makedirs(case_name+"/ori/m", exist_ok=True)

# ディレクトリパスのリスト
cvs = ["/cv_0", "/cv_1", "/cv_2", "/cv_3", "/cv_4"]
dirs=["/train/f","/train/m","/validation/f","/validation/m",
      "/weight","/gradcam_t","/gradcam_v"]

# 各ディレクトリに対してディレクトリを作成
for cv in cvs:
    for dir in dirs:
        os.makedirs(case_name+cv+dir, exist_ok=True)

