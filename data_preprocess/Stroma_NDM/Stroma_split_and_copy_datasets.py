import os 
import shutil
join = os.path.join

path = '/vast/AI_team/dataset/seg_test_colon/2048/D'
path_list = sorted(next(os.walk(path))[1])

aim_path = '/vast/AI_team/sukmin/Datasets_for_inference/2022_colon_200x - N,D,M/D/images_2'
os.makedirs(aim_path, exist_ok=True)


for f in path_list:

    f_path = join(path, f)
    f_list = sorted(next(os.walk(f_path))[2])

    for case in f_list:
        print()
        print(case)
        input = join(f_path, case)
        output = join(aim_path, case)
        shutil.copyfile(input, output)
        
