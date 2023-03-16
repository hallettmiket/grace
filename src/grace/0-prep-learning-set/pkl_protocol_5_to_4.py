# labelbox seems to exporting a protocol 5 pkl file now; need to down grade it currently to 4
# this must be done in python > 3.7

import pandas as pd
import pickle as pck

train=pd.read_pickle("/home/data/refined/candescence/train-data/grace_tc/train_grace_tc.pkl")
val=pd.read_pickle("/home/data/refined/candescence/train-data/grace_tc/val_grace_tc.pkl")


filename_train="/home/data/refined/candescence/train-data/grace_tc/train_grace_tc.4.pkl"
with open(filename_train, 'wb') as pfile_train:
    pck.dump(train, pfile_train, protocol=4)

filename_val="/home/data/refined/candescence/train-data/grace_tc/val_grace_tc.4.pkl"
with open(filename_val, 'wb') as pfile_val:
    pck.dump(val, pfile_val, protocol=4)
