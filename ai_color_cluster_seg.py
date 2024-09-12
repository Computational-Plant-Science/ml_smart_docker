'''
Name: ai_color_cluster_seg.py

Version: 1.0

Summary: A machine learning model U2net and opencv based color clustering method hat performs object segmentation in a single shot
    
Author: Suxing Liu

Author-email: suxingliu@gmail.com

Created: 2023-06-01

USAGE:

    Default parameters: python3 ai_color_cluster_seg.py -p ~/example/ -ft png

    User defined parameters: python3 ai_color_cluster_seg.py -p ~/example/ -ft png -o ~/example/results/ -s lab -c 2 -min 500 -max 1000000 -pl 0

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
        




# gets the bounding boxes of contours and calculates the distance between two rectangles
def calculate_contour_distance(contour1, contour2): 
    
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)


# using numpy.concatenate because each contour is just a numpy array of points
def merge_contours(contour1, contour2):
    
    return np.concatenate((contour1, contour2), axis=0)
    
    #return np.vstack([contour1, contour2])


#group contours such that one contour corresponds to one object.
#when some contours that belong to the same object are detected separately
def agglomerative_cluster(contours, threshold_distance=40.0):
    
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else: 
            break

    return current_contours



# segment foreground object using color clustering method
def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters):
    

    #image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    #cl = ColorLabeler()
    
    # Change image color space, if necessary.
    colorSpace = args_colorspace.lower()

    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
    else:
        colorSpace = 'bgr'  # set for file naming purposes

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:,:,channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)
            
    (height, width, n_channel) = image.shape
    
    if n_channel > 1:
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
 
    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    
    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    
    # define number of cluster, at lease 2 cluster including background
    numClusters = max(2, args_num_clusters)
    
    # clustering method
    kmeans = KMeans(n_clusters = numClusters, n_init = 40, max_iter = 500).fit(reshaped)
    
    # get lables 
    pred_label = kmeans.labels_
    
    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)],key = lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i
    
    ret, thresh = cv2.threshold(kmeansImage,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    '''
    if np.count_nonzero(thresh) > 0:
        
        thresh_cleaned = clear_border(thresh)
    else:
        thresh_cleaned = thresh
    '''
    
    thresh_cleaned = thresh
    
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(thresh_cleaned, connectivity = 8)

    '''
    # stats[0], centroids[0] are for the background label. ignore
    # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
    
    # extract the connected component statistics for the current label
    sizes = stats[1:, cv2.CC_STAT_AREA]
    Coord_left = stats[1:, cv2.CC_STAT_LEFT]
    Coord_top = stats[1:, cv2.CC_STAT_TOP]
    Coord_width = stats[1:, cv2.CC_STAT_WIDTH]
    Coord_height = stats[1:, cv2.CC_STAT_HEIGHT]
    Coord_centroids = np.delete(centroids,(0), axis=0)
    

    
    #print("Coord_centroids {}\n".format(centroids[1][1]))
    
    #print("[width, height] {} {}\n".format(width, height))
    
    numLabels = numLabels - 1
    '''

    
    
    ################################################################################################
    '''
    if width*height < 1000000:
        max_size = width*height
    else:
        max_size = args_max_size
    '''

    min_size = args_min_size
    max_size = min(width*height, args_max_size)

    # initialize an output mask
    mask = np.zeros(gray.shape, dtype="uint8")
    
    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
    # extract the connected component statistics for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
    
        
        # ensure the width, height, and area are all neither too small
        # nor too big
        keepWidth = w > 0 and w < 6000
        keepHeight = h > 0 and h < 4000
        keepArea = area > min_size and area < max_size
        
        #if all((keepWidth, keepHeight, keepArea)):
        # ensure the connected component we are examining passes all three tests
        #if all((keepWidth, keepHeight, keepArea)):
        if keepArea:
        # construct a mask for the current connected component and
        # then take the bitwise OR with the mask
            print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
            

    img_thresh = mask
    
    
    ###################################################################################################
    size_kernel = 5
    
    #if mask contains mutiple non-connected parts, combine them into one. 
    (contours, hier) = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        
        print("mask contains mutiple non-connected parts, combine them into one\n")
        
        kernel = np.ones((size_kernel,size_kernel), np.uint8)

        dilation = cv2.dilate(img_thresh.copy(), kernel, iterations = 1)
        
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        
        img_thresh = closing
    
    
    '''
    (height, width, n_channel) = img_thresh.shape
    
    if n_channel > 0:
        
        # convert the mask image to gray format
        img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)
    '''
    return img_thresh

    
'''
# compute medial axis from the mask of image
def medial_axis_image(thresh):
    
    #convert an image from OpenCV to skimage
    thresh_sk = img_as_float(thresh)

    image_bw = img_as_bool((thresh_sk))
    
    image_medial_axis = medial_axis(image_bw)
    
    return image_medial_axis


# compute the skeleton from the mask of image
def skeleton_bw(thresh):

    # Convert mask to boolean image, rather than 0 and 255 for skimage to use it
    
    #convert an image from OpenCV to skimage
    thresh_sk = img_as_float(thresh)

    image_bw = img_as_bool((thresh_sk))

    #skeleton = morphology.skeletonize(image_bw)
    
    skeleton = morphology.thin(image_bw)
    
    skeleton_img = skeleton.astype(np.uint8) * 255



    return skeleton_img, skeleton

'''

# compute percentage as two decimals value
def percentage(part, whole):
  
  #percentage = "{:.0%}".format(float(part)/float(whole))
  
  percentage = "{:.2f}".format(float(part)/float(whole))
  
  return str(percentage)


'''
# convert image from RGB to LAB color space
def image_BRG2LAB(image_file):

   # extarct path and name of the image file
    abs_path = os.path.abspath(image_file)
    
    filename, file_extension = os.path.splitext(abs_path)
    
    # extract the base name 
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # get the image file name
    image_file_name = Path(image_file).name
    
    print("Converting image {0} from RGB to LAB color space\n".format(str(image_file_name)))
    
    
    # load the input image 
    image = cv2.imread(image_file)
    
    # change to RGB space
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #plt.imshow(image_RGB)
    #plt.show()
    
    # get pixel color
    pixel_colors = image_RGB.reshape((np.shape(image_RGB)[0]*np.shape(image_RGB)[1], 3))
    
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    
    norm.autoscale(pixel_colors)
    
    pixel_colors = norm(pixel_colors).tolist()
    
    #pixel_colors_array = np.asarray(pixel_colors)
    
    #pixel_colors = pixel_colors.ravel()
    
    # change to lab space
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB )
    
   
    (L_chanel, A_chanel, B_chanel) = cv2.split(image_LAB)
    

    ######################################################################
   
    
    fig = plt.figure(figsize=(8.0, 6.0))
    
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    axis.scatter(L_chanel.flatten(), A_chanel.flatten(), B_chanel.flatten(), facecolors = pixel_colors, marker = ".")
    axis.set_xlabel("L:ightness")
    axis.set_ylabel("A:red/green coordinate")
    axis.set_zlabel("B:yellow/blue coordinate")
    
    # save segmentation result
    result_file = (save_path + base_name + '_lab' + file_extension)
    
    plt.savefig(result_file, bbox_inches = 'tight', dpi = 1000)
    
'''


# compute the size and shape info of the foreground
def comp_external_contour(orig, thresh):
    
    #find contours and get the external one
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    thresh_merged = thresh
    
    if len(contours) > 3:
        
        #####################################################################################
        print("Merging contours...\n")
        
        # using agglomerative clustering algorithm to group contours belonging to same object
        contours_list = [element for element in contours]

        # merge adjacent contours with distance threshold
        gp_contours = agglomerative_cluster(contours_list, threshold_distance = 50.0)

        #test_mask = np.zeros([height, width], dtype="uint8")

        thresh_merged = cv2.drawContours(thresh_merged, gp_contours, -1,255,-1)

        #define result path for labeled images
        #result_img_path = file_path + 'test_mask.png'
        #cv2.imwrite(result_img_path, test_mask)
        
        ################################################################################
        #find contours and get the external one
        contours, hier = cv2.findContours(thresh_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    else:
        
        #####################################################################################
        print("No need to merge contours...\n")
    
    
    
    
    
    # sort the contours based on area from largest to smallest
    contours_sorted = sorted(contours, key = cv2.contourArea, reverse = True)
    
    # initialize parameters
    area_c_cmax = 0
   
    img_height, img_width, img_channels = orig.shape
   
    index = 1
    
    trait_img = orig.copy()
    
    area = 0
    
    solidity = 0
    
    w=h=0
    
    
    for index, c in enumerate(contours_sorted):
        
    #for c in contours:
        if index < 1:
    
            #get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            
            if w>img_width*0.01 and h>img_height*0.01:
                
                trait_img = cv2.drawContours(orig, contours, -1, (255, 255, 0), 1)
        
                # draw a green rectangle to visualize the bounding rect
                roi = orig[y:y+h, x:x+w]
                
                print("ROI {} detected ...\n".format(index))
                #result_file = (save_path +  str(index) + file_extension)
                #cv2.imwrite(result_file, roi)
                
                trait_img = cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 0), 3)
                
                index+= 1

                '''
                #get the min area rect
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # convert all coordinates floating point values to int
                box = np.int0(box)
                #draw a red 'nghien' rectangle
                trait_img = cv2.drawContours(orig, [box], 0, (0, 0, 255))
                '''
                 # get convex hull
                hull = cv2.convexHull(c)
                # draw it in red color
                trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255), 3)
                
                '''
                # calculate epsilon base on contour's perimeter
                # contour's perimeter is returned by cv2.arcLength
                epsilon = 0.01 * cv2.arcLength(c, True)
                # get approx polygons
                approx = cv2.approxPolyDP(c, epsilon, True)
                # draw approx polygons
                trait_img = cv2.drawContours(orig, [approx], -1, (0, 255, 0), 1)
             
                # hull is convex shape as a polygon
                hull = cv2.convexHull(c)
                trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255))
                '''
                
                '''
                #get the min enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = (int(x), int(y))
                radius = int(radius)
                # and draw the circle in blue
                trait_img = cv2.circle(orig, center, radius, (255, 0, 0), 2)
                '''
                
                area = cv2.contourArea(c)
                print("Leaf area = {0:.2f}... \n".format(area))
                
                
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area
                print("solidity = {0:.2f}... \n".format(solidity))
                
                extLeft = tuple(c[c[:,:,0].argmin()][0])
                extRight = tuple(c[c[:,:,0].argmax()][0])
                extTop = tuple(c[c[:,:,1].argmin()][0])
                extBot = tuple(c[c[:,:,1].argmax()][0])
                
                trait_img = cv2.circle(orig, extLeft, 3, (255, 0, 0), -1)
                trait_img = cv2.circle(orig, extRight, 3, (255, 0, 0), -1)
                trait_img = cv2.circle(orig, extTop, 3, (255, 0, 0), -1)
                trait_img = cv2.circle(orig, extBot, 3, (255, 0, 0), -1)
                
                max_width = dist.euclidean(extLeft, extRight)
                max_height = dist.euclidean(extTop, extBot)
                
                if max_width > max_height:
                    trait_img = cv2.line(orig, extLeft, extRight, (0,255,0), 2)
                else:
                    trait_img = cv2.line(orig, extTop, extBot, (0,255,0), 2)
                    

                print("Width and height are {0:.2f},{1:.2f}... \n".format(w, h))
        
        
            
    return trait_img, area, solidity, w, h
    
    
    
    


# rgb to hex conversion
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    
# rgb to float conversion
def RGB2FLOAT(color):
    return "{:.2f}{:.2f}{:.2f}".format(int(color[0]/255.0), int(color[1]/255.0), int(color[2]/255.0))


# get color map from index
def get_cmap(n, name = 'hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.get_cmap(name, n)
    
    #import matplotlib.cm
    #return matplotlib.cm.get_cmap(name, n)
    


# cluster colors in the masked image
def color_region(image, mask, save_path, num_clusters):
    
    # read the image
    # get image width and height
    (h, w) = image.shape[:2]

    #apply the mask to get the segmentation of plant
    masked_image_ori = cv2.bitwise_and(image, image, mask = mask)
    
    # convert to RGB
    image_RGB = cv2.cvtColor(masked_image_ori, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image_RGB.reshape((-1, 3))
    
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    compactness, labels, (centers) = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels_flat = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels_flat]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image_RGB.shape)

    segmented_image_BRG = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    
    if args["debug"] == 1:

        #define result path for labeled images
        result_img_path = save_path + 'masked.png'
        cv2.imwrite(result_img_path, masked_image_ori)
        
        #define result path for labeled images
        result_img_path = save_path + 'clustered.png'
        cv2.imwrite(result_img_path, segmented_image_BRG)

    #define result path for labeled images
    #result_img_path = save_path + 'clustered.png'
    #cv2.imwrite(result_img_path, segmented_image_BRG)


    #Show only one chosen cluster 
    #masked_image = np.copy(image)
    masked_image = np.zeros_like(image_RGB)

    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to render
    #cluster = 2

    #cmap = get_cmap(num_clusters + 1)

    ####################################################################
    counts = Counter(labels_flat)

    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = centers

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    #rgb_colors = [RGB2FLOAT(ordered_colors[i]) for i in counts.keys()]

    #rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
    rgb_colors = [np.array(ordered_colors[i]).reshape(1, 3) for i in counts.keys()]

    index_bkg = [index for index in range(len(hex_colors)) if hex_colors[index] == '#000000']
   
    if len(index_bkg) > 0:
        
        #remove background color 
        del hex_colors[index_bkg[0]]
        del rgb_colors[index_bkg[0]]
        
        # Using dictionary comprehension to find list 
        # keys having value . 
        delete = [key for key in counts if key == index_bkg[0]] 
      
        # delete the key 
        for key in delete: del counts[key] 
    
    ########################################################################
    #compute color cluster ratio in percentage
    list_counts = list(counts.values())
    
    print("list_counts = {}\n".format(list_counts))
    
    color_ratio = []

    for value_counts, value_hex in zip(list_counts, hex_colors):

        #print(percentage(value, np.sum(list_counts)))

        color_ratio.append(percentage(value_counts, np.sum(list_counts)))

    
   
    return rgb_colors, counts, hex_colors, color_ratio




# compute the image brightness
def remove_character_string(str_input):
    
    return str_input.replace('#', '')


# compute the mean value of hex colors
def hex_mean_color(color_list):
    
    average_value = (int(remove_character_string(color1), 16) + int(remove_character_string(color2), 16) + int(remove_character_string(color3), 16) + int(remove_character_string(color4), 16))//4
       
    return hex(average_value)




# convert from RGB to LAB space,
# Convert it to LAB color space to access the luminous channel which is independent of colors.
def RGB2LAB(image, mask):
    
    # Make backup image
    image_rgb = image.copy()
    
    # apply object mask
    masked_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask = mask)
    
    # Convert color space to LAB space and extract L channel
    (L, A, B) = cv2.split(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2LAB))
    

    return masked_rgb, L, A, B
    

'''
# Max RGB filter 
def max_rgb_filter(image):
    
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)
    
    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
    
    
    # merge the channels back together and return the image
    return cv2.merge([B, G, R])
'''

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
    #roi_image = remove(roi_image).copy()

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
    # convert whole foreground object from RGB to LAB color space 
    '''
    (masked_rgb, L, A, B) = RGB2LAB(ROI_region.copy(), thresh)

    print("L_max = {} L_min = {}\n".format(L.max(), L.min()))
    print("A_max = {} A_min = {}\n".format(A.max(), A.min()))
    print("B_max = {} B_min = {}\n".format(B.max(), B.min()))
    
    
    
    result_img_path = save_path + 'masked_rgb.png'
    cv2.imwrite(result_img_path, masked_rgb)
    
    result_img_path = save_path + 'L.png'
    cv2.imwrite(result_img_path, L)
    
    result_img_path = save_path + 'A.png'
    cv2.imwrite(result_img_path, A)
    
    result_img_path = save_path + 'B.png'
    cv2.imwrite(result_img_path, B)
    '''
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
    #(trait_img, area, solidity, max_width, max_height) = comp_external_contour(ROI_region, thresh)


    #return thresh, masked_rgb
    
    return thresh_seg, masked_rgb_seg
        

    

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

            print("Start segmenting foreground object for image {} ...\n".format(file_path))

            # main pipeline to perform the segmentation based on u2net and color clustering
            (thresh, masked_rgb) = u2net_color_cluster(image_file)

            # save mask result image as png format
            #write_file(thresh, result_path, basename, '_mask.', 'png')

            # save masked result image as png format
            write_file(masked_rgb, result_path, basename, '_masked.', 'png')

            # store iteration end timestamp
            end = time.time()

            # show time of execution per iteration
            #print(f"Segmentation finished for: {filename}\tTime taken: {(end - start) * 10 ** 3:.03f}s !\n")

            print("Segmentation finished for: {} in --- {} seconds ---\n".format(filename, (end - start)))





    

    
