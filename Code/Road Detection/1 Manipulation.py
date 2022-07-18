# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pickle
import cv2
from tqdm import tqdm_notebook
import os
import json
import math
import h5py


from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
# %matplotlib inline

img_width = img_height = 256
num_channels = 3
root_path = "/content/data/"
mode = 'train'

def crop_and_save():
    """
    Imports images and crops to multiple sub images of a definite size inorder to preserve the resolution of the images and maximise the number of images that are available.
    """
    
    files = next(os.walk(root_path + mode + '/sat/'))[2]
    print('Total number of files =',len(files))
    
    for image_file in tqdm_notebook(files, total = len(files)):
       
   
        image_path = root_path+mode+'/sat/'+image_file
        image = cv2.imread(image_path)
        
        mask_path = root_path+mode+'/map/'+image_file[:-2]+'_'
        mask = cv2.imread(mask_path, 0)
        num_splits = math.floor((image.shape[0]*image.shape[1])/(img_width*img_height))
        counter = 0
        if mask is None:
          print("Can't load image, please check the path")
        
        
        for r in range(0, image.shape[0], img_height):
            for c in range(0, image.shape[1], img_width):
                counter += 1
                blank_image = np.zeros((img_height ,img_width, 3), dtype = int)
                blank_mask = np.zeros((img_height ,img_width), dtype = int)
                
                new_image_name = root_path+mode+'/sat2/' + str(counter) + '_' + image_file
                new_mask_name = root_path+mode+'/map2/' + str(counter) + '_' + image_file
                new_image = np.array(image[r:r+img_height, c:c+img_width,:])
                new_mask = np.array(mask[r:r+img_height, c:c+img_width])
            

                blank_image[:new_image.shape[0], :new_image.shape[1], :] += new_image
                blank_mask[:new_image.shape[0], :new_image.shape[1]] += new_mask

                
                cv2.imwrite(new_image_name, blank_image)
                cv2.imwrite(new_mask_name, blank_mask)
        
crop_and_save()

all_masks = []
all_images = []
def compress_images():
    """
    Imports images and respective masks and exports all of them into a h5py file.
    """
    
    global all_images, all_masks
    rej_count = 0
    counter = 0
    
    files = next(os.walk(root_path + mode + '/sat2/'))[2]
    print('Total number of files =',len(files))
 

    for image_file in tqdm_notebook(files, total = len(files)):
        
        counter += 1
           
        
        image_path = root_path+mode+'/sat2/'+image_file
        if not os.path.exists(image_path): continue    
        image = cv2.imread(image_path)
        imshow(image)
        
    
    
        mask_path = root_path+mode+'/map2/'+image_file
        mask = cv2.imread(mask_path, 0)
                  
        
        if (len(np.unique(mask)) == 1):
            rej_count += 1
            continue
        
        
        all_images.append(image)
        all_masks.append(mask)   
        
    
    all_images = np.asarray(all_images)
    all_masks = np.asarray(all_masks)
    
    
    print('{} images were rejected.'.format(rej_count))
    print("Shape of Train Images =", all_images.shape)
    print("Shape of Train Labels =", all_masks.shape)
    print("Memory size of Image array = ", all_images.nbytes)

    
    with h5py.File('/content/Data/'+'images.h5py', 'w') as hf: 
        hf.create_dataset("all_images",  data=all_images)
    
    with h5py.File('/content/Data/'+'masks.h5py', 'w') as hf:
       hf.create_dataset("all_masks",  data=all_masks)

    print("Data has been successfully exported.")

compress_images()

print("Shape of Train Images =", all_images.shape)
print("Shape of Train Labels =", all_masks.shape)