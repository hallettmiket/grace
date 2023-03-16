#    python ~/repo/deepmicroscopy/src/8-curriculum/mrc_40.py 
#from subprocess import Popen
import os
import sys

ROOT="/home/hallett/repo/candescence/"
OUTPUT="/home/data/refined/candescence/output/"
experiment_folder = ROOT + "/src/3-curriculum/configs/"
TOOL="/home/data/analysis-tools/""

resume = False # flip to true if you want to restart after a crash
current = "white"

exp = "0"  # experiment number
gpu = "3"
lr = "0.001"
momentum = "0.9"
decay = "0.0001"
total_epochs={"white":100,"opaque":100,"gray":100,"shmoo":100,"pseudohyphae":100  ,"hyphae":100}
grades = {"white":1, "opaque":2,"gray":3,"shmoo":4,"pseudohyphae":5  ,"hyphae":6}
order_names = ["white","opaque","gray","shmoo","pseudohyphae","hyphae"]
freeze = {"white":1,"opaque":2,"gray":2,"shmoo":2,"pseudohyphae":2  ,"hyphae":2}


if (resume==True):
    order_names_p = order_names[grades[current]-1:len(order_names)]       
        
    for j in order_names_p:
        config_path = ROOT + "src/8-curriculum/configs/" + j + "/exp" + exp + ".py"
        output_folder = OUTPUT + "/exp" + exp 
        full_command = "python " + TOOL + "mmdetection/tools/train.py" + " "\
            + config_path + " " +\
            "--work-dir=" + output_folder 
        full_command = full_command + " --resume-from=" + OUTPUT + "exp" + exp + "/latest.pth" 
        full_command = full_command + " " + "--gpu-ids=" + gpu
        try:
            os.system(full_command)
            try:
                os.mkdir(output_folder + "/" + j)
            except OSError:
                print ("Creation of the directory %s failed" % j)
            os.system("cp "+ output_folder + "/*" + " " +  output_folder + "/" + j)
        except:
            print("\n\nSystem failed with grade ", j)
      
else:

    for i in order_names:
        experiment_file = experiment_folder + i + "/exp" + exp + ".py"
        f = open(experiment_file, "w")
        f.write("\n_base_=[ 'config" + str(freeze[i]) + ".py' ]\n")
        f.write("\ntotal_epochs = " + str(total_epochs[i]) + "\n")
        f.write("\noptimizer = dict(lr=" + lr + ", momentum=" + momentum + ", weight_decay=" + decay + ", paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))\n")
        if (i=="white"):
            f.write("\nload_from = None\n")
        else:
            f.write("\nload_from = " + OUTPUT + "exp" + exp +  "/latest.pth'\n")
        f.close()

  

    for i in order_names:
        config_path = experiment_folder + i + "/exp" + exp + ".py"
        output_folder = OUTPUT + "exp" + exp 
        full_command = "python " + TOOL + "mmdetection/tools/train.py"  + " "\
            + config_path + " " +\
            "--work-dir=" + output_folder
        full_command = full_command + " " + "--gpu-ids=" + gpu
        try:
            os.system(full_command)
            try:
                os.mkdir(output_folder + "/" + i)
            except OSError:
                print ("Creation of the directory %s failed" % i)
            os.system("cp "+ output_folder + "/*" + " " +  output_folder + "/" + i)
        except:
            print("\n\nSystem failed with grade ", i)
             
   
# os.system("curl -X POST  https://maker.ifttt.com/trigger/{finished}/with/key/<yourkeyhere>?value1=" + exp)  # sends to my phone

