'''
Name: grid_seg.py

Version: 1.0

Summary: A machine learning model U2net and opencv based color clustering method hat performs object segmentation in a single shot
    
Author: Suxing Liu

Author-email: suxingliu@gmail.com

Created: 2023-06-01

USAGE:

    Default parameters: python3 grid_seg.py -p ~/example/ -ft png


PARAMETERS:
    ("-p", "--path", dest = "path", type = str, required = True,    help = "path to image file")
    ("-ft", "--filetype", dest = "filetype", type = str, required = False, default='jpg,png', help = "Image filetype")
    ("-o", "--output_path", dest = "output_path", type = str, required = False,    help = "result path")
    ('-s', '--color_space', dest = "color_space", type = str, required = False, default ='lab', help='Color space to use: BGR, HSV, Lab, YCrCb (YCC)')
    ('-c', '--channels', dest = "channels", type = str, required = False, default='2', help='Channel indices to use for clustering, where 0 is the first channel,'
                                                                       + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" '
                                                                       + 'selects channels B and R. (default "all")')
    ('-min', '--min_size', dest = "min_size", type = int, required = False, default = 500,  help = 'min size of object to be segmented.')
    ('-max', '--max_size', dest = "max_size", type = int, required = False, default = 1000000,  help = 'max size of object to be segmented.')
    ('-pl', '--parallel', dest = "parallel", type = int, required = False, default = 0,  help = 'Whether using parallel processing or loop processing, 0: Loop, 1: Parallel')

INPUT:
    Image file

OUTPUT:
    Segmentation mask and masked foreground image


'''




# import the necessary packages
import os
import glob
import pathlib
from pathlib import Path

from collections import Counter
from collections import OrderedDict

from sklearn.cluster import KMeans

#from skimage.morphology import medial_axis
#from skimage import img_as_float, img_as_ubyte, img_as_bool, img_as_int
#from skimage import morphology
#from skimage.segmentation import clear_border, watershed

from scipy.spatial import distance as dist
import PIL
from PIL import Image

import numpy as np
import argparse
import cv2
import imutils

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from rembg import remove

import time

MBFACTOR = float(1<<20)



# generate folder to store the output results
def mkdir(path):
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        
        print ("{} path exists!\n".format(path))
        return False
        

# ofind middle points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)




# divide the image into grids and get properities of each block
def grid_seg(input_img, nRows, mCols):
    
    
    img_height, img_width, img_channels = input_img.shape

    # Dimensions of the image
    sizeX = img_width
    sizeY = img_height


    blocks = []
    
    #offset = 40
    
    
    offset = 0
    
    blocks_id = []
    
    for i in range(0, nRows):
        
        for j in range(0, mCols):
            
            seg = input_img[int(i*sizeY/nRows) + offset : int(i*sizeY/nRows) + int(sizeY/nRows) - offset, int(j*sizeX/mCols) + offset : int(j*sizeX/mCols) + int(sizeX/mCols) - offset]
            
                        
            #result_file = (result_path +  str(i+1) + str(j+1) + '.png')
            
            #cv2.imwrite(result_file, seg)
            
            id_name = str(i+1) + str(j+1)
            
            blocks.append(seg)
            
            blocks_id.append(id_name)


    return blocks, blocks_id












# compute all the traits
def u2net_color_cluster(image_file):


    ################################################################################
    # load image data

    if args['filetype'] == 'CR2':
        pil_img = Image.open(image_file)

        cv2_img_arr = np.array(pil_img)
        image = cv2.cvtColor(cv2_img_arr, cv2.COLOR_RGB2BGR)

    else:
        image = cv2.imread(image_file)
    
    # make backup image
    orig = image.copy()
    
    # get the dimension of the image
    img_height, img_width, img_channels = orig.shape

    file_size = int(os.path.getsize(image_file)/MBFACTOR)
    
    ################################################################################
    # output image file info
    if image is not None:
        
        print("Plant object segmentation using u2net and color clustering method... \n")
        
        print("Image file size: {} MB, dimension: {} X {}, channels : {}\n".format(str(file_size), img_height, img_width, img_channels))
    

    ##################################################################################

    ROI_region = image.copy()
    
    #orig = sticker_crop_img.copy()
    #result_img_path = result_path + 'ROI_region.png'
    #cv2.imwrite(result_img_path, ROI_region)
    
    ##########################################################################
    #Plant region detection (defined as ROI_region)

    roi_image = ROI_region.copy()
    

    ###################################################################################
    # PhotoRoom Remove Background API
    
    # AI pre-trained model to segment plant object, test function
    roi_image = remove(roi_image).copy()

    #result_img_path = result_path + 'roi_image.png'
    #cv2.imwrite(result_img_path, roi_image)
    
    ######################################################################################
    #orig = roi_image.copy()
    n_cluster = 2
    
    '''
    #color clustering based plant object segmentation, return plant object mask
    thresh = color_cluster_seg(roi_image, args_colorspace, args_channels, n_cluster)
    
    masked_rgb = cv2.bitwise_and(ROI_region, ROI_region, mask = thresh)
    
    #masked_rgb_seg = remove(masked_rgb, bgcolor=(0, 0, 0, 255)).copy()
    
    thresh_seg = remove(masked_rgb, only_mask = True).copy()
    
    thresh_seg = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #mask_combined = thresh_seg & thresh
    
    #print(thresh_seg.shape)
    
    #print(thresh.shape)
    '''
    
    thresh_seg = remove(roi_image, only_mask = True).copy()
    
    
    #thresh_seg = cv2.threshold(thresh_seg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    

    
    
    #result_img_path = result_path + 'masked_rgb_seg.png'
    #cv2.imwrite(result_img_path, masked_rgb_seg)
    
    #result_img_path = result_path + 'thresh_seg.png'
    #cv2.imwrite(result_img_path, thresh_seg)
    
    
    #result_img_path = result_path + 'thresh.png'
    #cv2.imwrite(result_img_path, thresh)
    
    ######################################################################################
    # combine mask
    #mask_combined = thresh_seg & thresh
    
    
    #result_img_path = result_path + 'mask_combined.png'
    #cv2.imwrite(result_img_path, mask_combined)
    
    
    masked_rgb_seg = cv2.bitwise_and(roi_image, roi_image, mask = thresh_seg)
    
    
    
    
    
    
    #########################################################################################################
    #number of rows
    nRows = 4
    # Number of columns
    mCols = 5
    

 

    #########################################################################################################

    #find contours and get the external one
    contours, hier = cv2.findContours(thresh_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort the contours based on area from largest to smallest
    contours_sorted = sorted(contours, key = cv2.contourArea, reverse = True)
    
    c_max = contours_sorted[0]
    
    
    #get the bounding rect
    (x, y, w, h) = cv2.boundingRect(c_max)
            
    
    #masked_mask_image = cv2.bitwise_and(mask_image, mask_image, mask = thresh_seg)
    
    roi = mask_image[y:y+h, x:x+w]
    
    print(orig.shape)
    
    print(mask_image.shape)
    
    
    # draw a green rectangle to visualize the bounding rect
    #roi = orig[y:y+h, x:x+w]

    #bk_img = thresh_seg.copy()
    
    #result_file = (result_path +  str(index) + file_extension)
    #cv2.imwrite(result_file, roi)

    #trait_img = cv2.rectangle(bk_img, (x, y), (x+w, y+h), (255, 255, 0), 3)


    
    (blocks, blocks_id) = grid_seg(roi, nRows, mCols)




    ##########################################################################################################
    
    # apply object mask
    #masked_rgb = cv2.bitwise_and(ROI_region, ROI_region, mask = thresh)

    ##########################################################################################################
    # color clustering using pre-defined color cluster value by user
    
    #print("number of cluster: {}\n".format(args_num_clusters))
    
    #color clustering of masked image
    #(rgb_colors, counts, hex_colors, color_ratio) = color_region(ROI_region, thresh, save_path, args_num_clusters)


    ###########################################################################################################
    #compute external contour, shape info  
    #(trait_img, area, solidity, max_width, max_height) = comp_external_contour(masked_rgb_seg, thresh_seg)


    #return thresh_seg, masked_rgb
    
    return thresh_seg, masked_rgb_seg, blocks, blocks_id
        

    

# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')
    
    return file_path, filename, basename



# save result files
def write_file(imagearray, result_path, base_name, addition, ext):

    # save segmentation result
    result_file = (result_path + base_name + addition + ext)
    
    #print(result_file)
    
    cv2.imwrite(result_file, imagearray)
    
    # check saved file
    if os.path.exists(result_file):
        print("Result file was saved at {0}\n".format(result_file))

    else:
        print("Result file writing failed!\n")
    

def batch_process(image_file):

    (file_path, filename, basename) = get_file_info(image_file)

    print("Segment foreground object for image file {} ...\n".format(file_path, filename, basename))

    # main pipeline to perform the segmentation based on u2net and color clustering
    (thresh, masked_rgb) = u2net_color_cluster(image_file)

    # save mask result image as png format
    # write_file(thresh, result_path, basename, '_mask.', 'png')

    # save masked result image as png format
    write_file(masked_rgb, result_path, basename, '_masked.', 'png')




if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", dest = "path", type = str, required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default='jpg,png', help = "Image filetype")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False,    help = "result path")
    ap.add_argument('-s', '--color_space', dest = "color_space", type = str, required = False, default ='lab', help='Color space to use: BGR, HSV, Lab, YCrCb (YCC)')
    ap.add_argument('-c', '--channels', dest = "channels", type = str, required = False, default='2', help='Channel indices to use for clustering, where 0 is the first channel,' 
                                                                       + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" ' 
                                                                       + 'selects channels B and R. (default "all")')
    #ap.add_argument('-n', '--num_clusters', dest = "num_clusters", type = int, required = False, default = 4,  help = 'Number of clusters for K-means clustering (default 2, min 2).')
    ap.add_argument('-min', '--min_size', dest = "min_size", type = int, required = False, default = 500,  help = 'min size of object to be segmented.')
    ap.add_argument('-max', '--max_size', dest = "max_size", type = int, required = False, default = 1000000,  help = 'max size of object to be segmented.')
    ap.add_argument('-pl', '--parallel', dest = "parallel", type = int, required = False, default = 0,  help = 'Whether using parallel processing or loop processing, 0: Loop, 1: Parallel')
    args = vars(ap.parse_args())
    

    
    # setup input and output file paths

    file_path = args["path"]
    ext = args['filetype']

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    
    ########################################################################
    '''
    parent_dir_file = os.path.dirname(file_path)
    
    print(f"Parent directory of '{file_path}': {parent_dir_file}")
    
    mask_path = parent_dir_file + "/masks/" + filetype
    
    print("mask_path: {}\n".format(mask_path))
    
    mask_imgList = sorted(glob.glob(mask_path))
    
    #print(mask_imgList)
    '''

    ###########################################################################
    # result file path
    result_path = args["output_path"] if args["output_path"] is not None else file_path

    result_path = os.path.join(result_path, '')
    
    # print out result path
    print("results_folder: {}\n".format(result_path))

    #########################################################################
    # scan the folder to remove the 0 size image
    for image_id, image_file in enumerate(imgList):
        try:
            image = Image.open(image_file)
        except PIL.UnidentifiedImageError as e:
            print(f"Error in file {image_file}: {e}")
            os.remove(image_file)
            print(f"Removed file {image_file}")

    ############################################################################
    #accquire image file list after remove error images
    imgList = sorted(glob.glob(image_file_path))


    ########################################################################
    # parameters
    args_min_size = args['min_size']
    args_max_size = args['max_size']

    args_colorspace = args['color_space']
    args_channels = args['channels']
    #args_num_clusters = args['num_clusters']

    args_parallel = args['parallel']




    if args_parallel == 1:
        # Parallel processing
        #################################################################################
        import psutil
        from multiprocessing import Pool
        from contextlib import closing

        # parallel processing
        # get cpu number for parallel processing
        agents = psutil.cpu_count() - 2

        print("Using {0} cores to perform parallel processing... \n".format(int(agents)))

        # Create a pool of processes. By default, one is created for each CPU in the machine.
        with closing(Pool(processes=agents)) as pool:
            result = pool.map(batch_process, imgList)
            pool.terminate()

    else:

        #########################################################################
        # analysis pipeline
        # loop execute

        for image_id, image_file in enumerate(imgList):
            # store iteration start timestamp
            start = time.time()

            (file_path, filename, basename) = get_file_info(image_file)

            print("Start segmenting foreground object for image {} ...\n".format(basename))
            
            

            # main pipeline to perform the segmentation based on u2net and color clustering
            #(thresh, masked_rgb, blocks, blocks_id) = u2net_color_cluster(image_file)
            
            
            ###############################################################################
            # direct segemtation using even grid
            image = cv2.imread(image_file)
    
            # make backup image
            orig = image.copy()
            
            # get the dimension of the image
            img_height, img_width, img_channels = orig.shape
            
            
            #number of rows
            nRows = 4
            # Number of columns
            mCols = 5
            
            (blocks, blocks_id) = grid_seg(orig, nRows, mCols)
            
            ########################################################################3
            # save mask result image as png format
            #write_file(thresh, result_path, basename, '_mask.', 'png')

            # save masked result image as png format
            #write_file(masked_rgb, result_path, basename, '_masked.', 'png')


            # save masked result image as png format
            #write_file(trait_img, result_path, basename, '_trait_img.', 'png')
            
            
            mkpath = file_path + '/' + basename
            mkdir(mkpath)
            result_path_current = mkpath + '/'
            
            
            for i, (block, id_name) in enumerate(zip(blocks, blocks_id)):

                id_str = ("{:02}".format(i))

                # save masked result image as png format
                write_file(block, result_path_current, id_name, '.', 'png')
            
            
            
            
            
            
            
            
            
            # store iteration end timestamp
            end = time.time()

            # show time of execution per iteration
            #print(f"Segmentation finished for: {filename}\tTime taken: {(end - start) * 10 ** 3:.03f}s !\n")

            print("Segmentation finished for: {} in --- {} seconds ---\n".format(filename, (end - start)))





    

    
