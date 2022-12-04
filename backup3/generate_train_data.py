import glob


path=r"/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/train/"
grids_path = path
grids_list = glob.glob(grids_path + "*_grids*pkl")
label_list = glob.glob(grids_path + "*_label*pkl")
data_test = []

for label_path in label_list:
        for grids_path in grids_list:
            if "_".join(grids_path.split('/')[-1].split('_')[0:2]) == \
               "_".join(label_path.split('/')[-1].split('_')[0:2]):
                data_test.append([grids_path, label_path])
        print(len(data_test))
        

import random
random.shuffle(data_test)
random.shuffle(data_test)
random.shuffle(data_test)        

import os
import pickle
os.chdir("/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/")
with open("train_data_2.pkl",'wb') as f: 
        pickle.dump(data_test, f)
