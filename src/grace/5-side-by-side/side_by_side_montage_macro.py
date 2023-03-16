from evaluator_macro import *
from glob import glob
import pandas as pd
from pathlib import Path
import numpy as np
import os


thresh = 0.2

CANDESCENCE = "/home/data/refined/candescence/"
config_file = CANDESCENCE + "production/models/macro_config.py"
network_file =  CANDESCENCE + "production/models/macro_model.pth"
output_dir = CANDESCENCE + "performance/grace_macro/side_by_side_" + str(thresh) + "/"
target_file = CANDESCENCE + "train-data/grace_macro/val_grace_macro.pkl"



eval = Evaluator(config_file, network_file, target_file, CANDESCENCE + "train-data/grace_macro/val/")


for ind in range(0,len(eval.filenames)):
    print("\nCurrent filename: %s" % eval.filenames[ind])
    annotations = eval.get_gts(eval.filenames[ind])
    new_image = mmcv.imshow_det_bboxes(
            eval.filenames[ind],
            annotations["ann"]["bboxes"],
            annotations["ann"]["labels"],
            class_names=eval.model.CLASSES,
            score_thr=0,
            wait_time=0,
            show=False,
            out_file=output_dir + "gt_" + str(ind) + ".png")
    eval.draw_silent_dts(eval.filenames[ind],thresh, output_dir + "pred_" + str(ind) + ".png")
    os.system( "montage " + output_dir + "gt_" + str(ind) + ".png " + output_dir + "pred_" + str(ind) + ".png -tile 2x1 -geometry +0+0 " + output_dir + "join" + str(ind) + ".png" )
    os.system( "rm " + output_dir + "gt_" + str(ind) + ".png" )
    os.system( "rm " + output_dir + "pred_" + str(ind) + ".png" )


