# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#v3.classification
#28/11/2018

dataname="brca1"

patch_size=256 #size of the tiles to extract and save in the database, must be >= to training size
test_set_size=.1 # what percentage of the dataset should be used as a held out validation/testing set
class_names=["Stroma","Tumor","Immune cells","Other"]

#-- relating to masks
downsampled=16
cut_off_percent=.9
kernel_size=patch_size//downsampled
max_number_samples=10

#-----Note---
#One should likely make sure that  (nrow+mirror_pad_size) mod patch_size == 0, where nrow is the number of rows after resizing
#so that no pixels are lost (any remainer is ignored)

# -


def random_subset(a, b, nitems):
    assert len(a) == len(b)
    idx = np.random.randint(0,len(a),nitems)
    return a[idx], b[idx]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

# +
import torch
import tables

import os,sys
import glob

import PIL
import numpy as np

import cv2
import matplotlib.pyplot as plt

import scipy.signal

from sklearn import model_selection
import sklearn.feature_extraction.image
import random

from tqdm.autonotebook import tqdm

#OpenSlide Path
openslide_path = 'C:\\research\\openslide\\bin'

os.environ['PATH'] = openslide_path + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
import openslide

random_seed_number = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the cross validation setup
random.seed(random_seed_number) # set the seed
print("random seed (note down for reproducibility): {random_seed_number}")
# -

img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just incase we need it later, 

# +
files=glob.glob('/home/marco/Documents/Lab/deepLearning/MaskLou/*.png')
#print(files)
bases=list(set(["_".join(f.split('_')[0:2]).replace("/home/marco/Documents/Lab/deepLearning/MaskLou\\","") for f in files]))

#create training and validation stages and split the files appropriately between them
phases={}
phases["train"],phases["val"]=next(iter(model_selection.ShuffleSplit(n_splits=1,test_size=test_set_size).split(bases)))
# -

basesImage=list(set(["_".join(f.split('_')[0:2]).replace("/home/marco/Documents/Lab/deepLearning/MaskLou\\","") for f in files]))

# +
storage={} #holder for future pytables

block_shape=np.array((patch_size,patch_size,3)) #block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d

filters=tables.Filters(complevel=6, complib='zlib') #we can also specify filters, such as compression, to improve storage speed

avg_filter = np.ones((kernel_size, kernel_size))

for phase in phases.keys(): #now for each of the phases, we'll loop through the files
    #print(phase)
    
    totals=np.zeros(len(class_names)) # we can to keep counts of all the classes in for in particular training, since we 
    
    hdf5_file = tables.open_file("./"+ dataname +"_"+ phase +".pytable", mode='w') #open the respective pytable
    storage["filenames"] = hdf5_file.create_earray(hdf5_file.root, 'filenames', filenameAtom, (0,)) #create the array for storage
    
    storage["imgs"]= hdf5_file.create_earray(hdf5_file.root, "imgs", img_dtype,  
                                              shape=np.append([0],block_shape), 
                                              chunkshape=np.append([1],block_shape),
                                              filters=filters)
    storage["labels"]= hdf5_file.create_earray(hdf5_file.root, "labels", img_dtype,  
                                              shape=[0], 
                                              chunkshape=[1],
                                              filters=filters)

    #tqdm progress bar management for cli
    for file_name in tqdm(phases[phase]): #now for each of the files
        print(bases[file_name])

        print(bases[file_name].replace("/home/marco/Documents/Lab/deepLearning/MaskLou/", "/data/brca1/mib1/"))
        osh  = openslide.OpenSlide(bases[file_name].replace("/home/marco/Documents/Lab/deepLearning/MaskLou/", "/data/brca1/mib1/")+".mrxs")

        #mask created by QPath
        masks_file_name = glob.glob('./masks/{bases[file_name]}*.png')

        for masks_file_name in tqdm(masks_file_name):

            classid=[idx for idx in range(len(class_names)) if class_names[idx] in masks_file_name][0]

            loaded_mask_image = cv2.imread(masks_file_name)
            loaded_mask_image = loaded_mask_image[:,:,0]
            normalized_mask_image = loaded_mask_image // 255

            loaded_mask_image = scipy.signal.fftconvolve( normalized_mask_image, avg_filter, mode='same')

            loaded_mask_image = loaded_mask_image >=(kernel_size**2)*cut_off_percent

            [rs,cs]=loaded_mask_image.nonzero()
            
            [rs,cs]=random_subset(rs,cs,min(max_number_samples,len(rs)))
            
            totals[classid]+=len(rs)

            for i, (r,c) in tqdm(enumerate(zip(rs,cs)),total =len(rs)):

                io = np.asarray(osh.read_region((c*downsampled-patch_size//2, r*downsampled-patch_size//2), 2, (patch_size, patch_size)))
                io = io[:, :, 0:3]  # remove alpha channel
                
                storage["imgs"].append(io[None,::])
                storage["labels"].append([int(classid)]) #add the filename to the storage array
                storage["filenames"].append([masks_file_name]) #add the filename to the storage array
            
        osh.close()
    #lastely, we should store the number of pixels
    npixels=hdf5_file.create_carray(hdf5_file.root, 'classsizes', tables.Atom.from_dtype(totals.dtype), totals.shape)
    npixels[:]=totals
    hdf5_file.close()

# +
#useful reference
#http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
