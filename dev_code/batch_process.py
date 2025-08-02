"""
Version: 1.5

Summary: Compute traits from a 3D model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 batch_process.py -i /input_path/ -o /output_path/


"""

import subprocess, os, glob
import sys
import argparse

import pathlib

import shutil


# execute script inside program
def execute_script(cmd_line):
    
    try:
        #print(cmd_line)
        #os.system(cmd_line)

        process = subprocess.getoutput(cmd_line)
        
        print(process)
        
        #process = subprocess.Popen(cmd_line, shell = True, stdout = subprocess.PIPE)
        #process.wait()
        #print (process.communicate())
        
    except OSError:
        
        print("Failed ...!\n")







# get file information from the file path using os for python 2.7
def get_fname(file_full_path):
    
    abs_path = os.path.abspath(file_full_path)

    filename= os.path.basename(abs_path)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    return filename, base_name




# get sub folders from a inout path for local test only
def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders




# get file information from the file path using pathon 3
def get_file_info(file_full_path):

    p = pathlib.Path(file_full_path)
    
    filename = p.name
    
    basename = p.stem


    file_path = p.parent.absolute()
    
    file_path = os.path.join(file_path, '')
    
    return file_path, filename, basename







# generate foloder to store the output results
def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #printpath + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        shutil.rmtree(path)
        os.makedirs(path)
        print("{} path exists!\n".format(path))
        return False
        



if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest = "input", type = str, required = True, help = "full path to 3D model file")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    args = vars(ap.parse_args())
    


    
    ####################################################################################
    # local test loop version
    
     # get input file path, name, base name.
    #(file_path, filename, basename) = get_file_info(args["input"])
    
    
    file_path = args["input"]
    
    
    
    output_path = args["output_path"] if args["output_path"] is not None else file_path
    
    subfolders = fast_scandir(file_path)
    
    for subfolder_id, subfolder_path in enumerate(subfolders):
    
        
        folder_name = os.path.basename(subfolder_path) 
        
        subfolder_path = os.path.join(subfolder_path, '')
        
       
        #print("Current sub folder path '{}'\n".format(subfolder_path))

        # python3 smart_dev.py -p ~/example/images_and_annotations/test/
        
        cmd_process = "python3 smart_rice.py -p " + subfolder_path
        
        #python3 batch_file_move.py -p ~/example/images_and_annotations/Tray031_select/PSI_Tray031_2015-12-14--12-54-06_top/ -r ~/example/images_and_annotations/Tray031_select_seg/
        
        #cmd_process = "python3 batch_file_move.py -p " + subfolder_path + " -r " + output_path
        
        #python3 mask_area.py -p ~/example/images_and_annotations/GT/031_mask/PSI_Tray031_2015-12-14--12-54-06_top_mask/
        
        
        #cmd_process = "python3 mask_area.py -p " + subfolder_path 
        

        print(cmd_process)

        execute_script(cmd_process)
    
    ##################################################################################
    
    

    
