"""
Version: 1.5

Summary: Automatic image brightness adjustment based on gamma correction method

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 color_checker_detect.py -p /input/ -ft jpg -gv 0.7


argument:
("-p", "--path", required = True,    help="path to image file")
("-ft", "--filetype", required=True,    help="Image filetype") 

"""

#!/usr/bin/python
# Standard Libraries

import os,fnmatch
import argparse
import shutil
import cv2
import pathlib


import numpy as np

import glob
import matplotlib.pyplot as plt

#import multiprocessing
#from multiprocessing import Pool
#from contextlib import closing

import resource
import imutils
from imutils import perspective

from skimage.segmentation import clear_border
from collections import Counter
from collections import OrderedDict

from scipy.spatial import distance as dist

from rembg import remove


# color label class
class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "dark skin": (115, 82, 68),
            "light skin": (194, 150, 130),
            "blue sky": (98, 122, 157),
            "foliage": (87, 108, 67),
            "blue flower": (133, 128, 177),
            "bluish green": (103, 189, 170),
            "orange": (214, 126, 44),
            "purplish blue": (8, 91, 166),
            "moderate red": (193, 90, 99),
            "purple": (94, 60, 108),
            "yellow green": (157, 188, 64),
            "orange yellow": (224, 163, 46),
            "blue": (56, 61, 150),
            "green": (70, 148, 73),
            "red": (175, 54, 60),
            "yellow": (231, 199, 31),
            "magneta": (187, 86, 149),
            "cyan": (8, 133, 161),
            "white": (243, 243, 242),
            "neutral 8": (200, 200, 200),
            "neutral 6.5": (160, 160, 160),
            "neutral 5": (122, 122, 121),
            "neutral 3.5": (85, 85, 85),
            "black": (52, 52, 52)})
        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
        #print("color_checker values:{}\n".format(self.lab))

    def label(self, image, c):
            # construct a mask for the contour, then compute the
            # average L*a*b* value for the masked region
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=2)
            mean = cv2.mean(image, mask=mask)[:3]

            # initialize the minimum distance found thus far
            minDist = (np.inf, None)
            # loop over the known L*a*b* color values
            for (i, row) in enumerate(self.lab):
                # compute the distance between the current L*a*b*
                # color value and the mean of the image
                d = dist.euclidean(row[0], mean)
                
                #print("mean = {0}, row = {1}, d = {2}, i = {3}\n".format(mean, row[0], d, i)) 
                
                # if the distance is smaller than the current distance,
                # then update the bookkeeping variable
                if d < minDist[0]:
                    minDist = (d, i)
            # return the name of the color with the smallest distance
            return self.colorNames[minDist[1]], mean

    def label_c(self, lab_color_value):
            # initialize the minimum distance found thus far
            minDist = (np.inf, None)
           
            # loop over the known L*a*b* color values
            for (i, row) in enumerate(self.lab):
                # compute the distance between the current L*a*b*
                # color value and the mean of the image
                d = dist.euclidean(row[0], lab_color_value)
                
                #print("mean = {0}, row = {1}, d = {2}, i = {3}\n".format(mean, row[0], d, i)) 
                
                # if the distance is smaller than the current distance,
                # then update the bookkeeping variable
                if d < minDist[0]:
                    minDist = (d, i)
            # return the name of the color with the smallest distance
            return self.colorNames[minDist[1]]







# create result folder
def mkdir(path):
    # import module
    #import os
 
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
        #print path+' path exists!'
        return False


# apply perspective transform to input image
def Perspective_Transform(roi_image_checker, roi_mask, masked_roi):
    
    
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(roi_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    
    checker_Cnt = None
    
    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            checker_Cnt = approx
            break
    
    # Apply PerspectiveTransform to the contour on the detected mask
    pts = checker_Cnt.reshape(4, 2)

    # 4 Point OpenCV getPerspective Transform  
    masked_roi_warped = four_point_transform(masked_roi, pts)
    
    
    return masked_roi_warped
    
    


# apply perspective transform to input image based on corner coordinates of a rect
def four_point_transform(image, pts):
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
        
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped


# sort corner coordinates of a rect
def order_points(pts):
    
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect


# ofind middle points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



# get block color value 
def block_color(masked_block, c_max):
    
    cl = ColorLabeler()

    lab_block = cv2.cvtColor(masked_block.copy(), cv2.COLOR_BGR2LAB)

    (color_name, color_value) = cl.label(lab_block, c_max)

    
    
    
    return color_name, color_value



# divide the image into grids and get properities of each block
def grid_seg(input_img, nRows, mCols):
    
    
    img_height, img_width, img_channels = input_img.shape

    # Dimensions of the image
    sizeX = img_width
    sizeY = img_height


    blocks = []
    
    blocks_overlay = input_img.copy()
    
    blocks_bk = input_img.copy()
    
    
    block_color_value = []
    block_width = []
    block_height = []
    block_area = []
            
    grid_area = []
    
    for i in range(0, nRows):
        
        for j in range(0, mCols):
            
            seg = input_img[int(i*sizeY/nRows):int(i*sizeY/nRows) + int(sizeY/nRows),int(j*sizeX/mCols):int(j*sizeX/mCols) + int(sizeX/mCols)]
            
            #blocks_overlay = cv2.putText(blocks_overlay, str("({0}{1})".format(i,j)), (int(j*sizeX/mCols) + + int(sizeY/nRows*0.5), int(i*sizeY/nRows) + int(sizeX/mCols*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
            blocks_overlay = cv2.putText(blocks_bk, str(len(blocks)), (int(j*sizeX/mCols) + + int(sizeY/nRows*0.5), int(i*sizeY/nRows) + int(sizeX/mCols*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)



            
            h, w, _channels = seg.shape
            
            seg_hsv = cv2.cvtColor(seg.copy(), cv2.COLOR_BGR2HSV)

            seg_gray = cv2.cvtColor(seg_hsv, cv2.COLOR_BGR2GRAY)

            thresh_block = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            #apply the mask to get the segmentation of block
            masked_block = cv2.bitwise_and(seg.copy(), seg.copy(), mask = thresh_block)
            
            # find contours in thresholded image, then grab the largest one
            cnts = cv2.findContours(thresh_block.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cnts = imutils.grab_contours(cnts)
            
            c_max = max(cnts, key=cv2.contourArea)
            
            area = cv2.contourArea(c_max)
            
            

            (color_name, color_value) = block_color(masked_block, c_max)

            
            
            #print("ID = ({}{}), color_name = {} color_value = {}\n".format(i, j, color_name, color_value))
            
            #cnt_block = cv2.drawContours(masked_block, [c_max], 0, (0, 0, 255), 2)
            
            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c_max)
            
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            
            box = np.array(box, dtype="int")
            
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding box
            #box = perspective.order_points(box)
            
            cnt_block = cv2.drawContours(masked_block, [box.astype("int")], 0, (0, 255, 0), 1)
            
            # loop over the original points and draw them
            for (x, y) in box:
                cnt_block = cv2.circle(masked_block, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            
            # draw the midpoints on the image
            cnt_block = cv2.circle(masked_block, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
            cnt_block = cv2.circle(masked_block, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
            cnt_block = cv2.circle(masked_block, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
            cnt_block = cv2.circle(masked_block, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)
            
            # draw lines between the midpoints
            cnt_block = cv2.line(masked_block, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1)
            cnt_block = cv2.line(masked_block, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1)
            
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
            # draw the object sizes on the image
            #cnt_block = cv2.putText(masked_block, "{}".format(dA), (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            #cnt_block = cv2.putText(masked_block, "{}".format(dB), (int(trbrX), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            
            #text_color = "{}".format(color_name)

            #cnt_block = cv2.putText(masked_block, text_color, (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            

            
            #print("dA = {}, dB = {} area = {} \n".format(dA,dB, area))



            blocks.append(cnt_block)

            block_color_value.append(color_value)
            
            block_width.append(dB)
            
            block_height.append(dA)
            
            block_area.append(area)
            
            grid_area.append(w*h)


    return blocks, blocks_overlay, block_color_value, block_width, block_height, block_area, grid_area





# Color checker detection
def color_checker_detection(roi_image_checker, result_path):
    
    
    #####################################################
    roi_image = remove(roi_image_checker).copy()

    # extract alpha channel
    alpha = roi_image[:, :, 3]

    # threshold alpha channel to get mask from alpha channel
    roi_mask_ori = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
    
    
    #########################################################
    k = np.ones((25, 25), np.uint8)  # Define 5x5 kernel
    
    #inv = cv2.bitwise_not(roi_mask_ori)
    
    roi_mask = cv2.erode(roi_mask_ori, k, 1)

    
    ################################################################################


    #apply the mask to get the segmentation of plant
    masked_roi = cv2.bitwise_and(roi_image_checker.copy(), roi_image_checker.copy(), mask = roi_mask)
    
    

    # projective transforamtion 
    masked_roi_warped = Perspective_Transform(roi_image_checker, roi_mask, masked_roi)

    
    
    ###################################################################
    # crop image to remove bottom extra 
    (r_height, r_width) = masked_roi_warped.shape[:2]
    
    print(r_height, r_width)

    # crop bottom part
    crop_h_ratio = 0.97
    
    start_y = 0  # Starting row (top)
    end_y = int(r_height*crop_h_ratio)    # Ending row (bottom)
    start_x = 0   # Starting column (left)
    end_x = r_width    # Ending column (right)
    
    # Crop the image
    cropped_image = masked_roi_warped[start_y:end_y, start_x:end_x]
    
    color_checker_detected = cropped_image

    
    ####################################################################
    #number of rows
    nRows = 4
    # Number of columns
    mCols = 6
    

    (blocks, blocks_overlay, block_color_value, block_width, block_height, block_area, grid_area) = grid_seg(cropped_image, nRows, mCols)
    
    
    # sort blocks based on area size
    
    index_keep = []
    
    for (i, block) in enumerate(blocks):

        print("ID = {}, color = {}, width = {}, height = {}, area = {}\n".format(i, block_color_value[i], block_width[i], block_height[i], block_area[i]))
        
        if block_area[i] > grid_area[i] *0.5:
            index_keep.append(i)
    
    
    
    selected_block_width = [block_width[i] for i in index_keep]
    
    selected_block_height = [block_height[i] for i in index_keep]
    
    
    print(selected_block_width)
    print(selected_block_height)
    
    if len(selected_block_width) > 0:
        average_width = np.mean(selected_block_width)
        print(f"The average width is: {average_width}")
    else:
        print("The list is empty, cannot calculate average.")
    
    if len(selected_block_height) > 0:
        average_height = np.mean(selected_block_height)
        print(f"The average height is: {average_height}")
    else:
        print("The list is empty, cannot calculate average.")

    
    
    return  average_width, average_height, roi_mask, masked_roi, masked_roi_warped, color_checker_detected, blocks, blocks_overlay
    


# save result files
def write_image_output(imagearray, result_path, base_name, addition, ext):

    # save segmentation result
    result_file = (result_path + base_name + addition + ext)
    
    #print(result_file)
    
    cv2.imwrite(result_file, imagearray)
    
    # check saved file
    if os.path.exists(result_file):
        print("Result file was saved at {0}\n".format(result_file))

    else:
        print("Result file writing failed!\n")



# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')
    
    return file_path, filename, basename





if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False,  default = 'jpg',  help = "image filetype")
    ap.add_argument("-gv", "--gamma_value", type = float, required = False,  default = 0.7,  help = "image filetype")
    args = vars(ap.parse_args())


    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    print((imgList))
    
        # Load the image
    
    image_save_path = file_path 
    
    # loop execute
    for image_id, image in enumerate(imgList):
        
        
        # get file information
        (file_path, filename, basename) = get_file_info(image)
        
        print("Processing image {} ... \n".format(file_path))
        
         # Load the image
        image = cv2.imread(image)

        #get size of image
        img_height, img_width = image.shape[:2]
        
        #####################################
    
        (average_width, average_height, roi_mask, masked_roi, masked_roi_warped, color_checker_detected, blocks, blocks_overlay) = color_checker_detection(image, image_save_path)
    

    
    
    
        ##############################################
        file_extension = '.' + ext

        print("image_save_path: {}\n".format(image_save_path))

        # save segmentation result
        write_image_output(roi_mask, image_save_path, basename, '_roi_mask', file_extension)

        write_image_output(masked_roi, image_save_path, basename, '_masked_roi', file_extension)

        write_image_output(masked_roi_warped, image_save_path, basename, '_masked_roi_warped', file_extension)

        write_image_output(color_checker_detected, image_save_path, basename, '_color_checker_detected', file_extension)
        
        write_image_output(blocks_overlay, image_save_path, basename, '_blocks_overlay', file_extension)
        
        
        
        for (i, block) in enumerate(blocks):

            result_file = (image_save_path +  str("{:02d}".format(i)) + '.png')

            cv2.imwrite(result_file, block)
        




   
    
    




