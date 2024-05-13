#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import nibabel as nib
import pandas as pd
join = os.path.join

path = '/vast/AI_team/sukmin/datasets/past_from_cv/_has_Task273_Urinary'
path_list = sorted(next(os.walk(path))[1])

l_1 = [] # ureter
l_2 = [] # bladder
l_3 = [] # kidney

array = [['케이스번호','ureter', 'bladder', 'kidney']]
a1 = []

for case in path_list:
    seg = nib.load(join(path, case, 'segmentation.nii.gz')).get_fdata()
    a1.append(case)

    if 1 in seg:
        l_1.append(case)
        a1.append(1)
    else:
        a1.append(0)


    if 2 in seg:
        l_2.append(case)
        a1.append(1)
    else:
        a1.append(0)


    if 3 in seg:
        l_3.append(case)
        a1.append(1)
    else:
        a1.append(0)

    array.append(a1)
    a1 = []
        
print("Total Ureter count : ", len(l_1))
print("Total bladder count : ", len(l_2))
print("Total kidney count : ", len(l_3))

# print("Ureter case : ", l_1)
# print("bladder case : ", l_2)
# print("kidney case : ", l_3)
        

df = pd.DataFrame(array)
print(df)
df.to_csv('ureter_dataset.csv')









# %%
