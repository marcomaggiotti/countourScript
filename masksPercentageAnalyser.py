import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


entries = os.listdir('/home/maorvelous/Documents/Lab/deepLearning/MaskLou/')
total_of_pixels = 88363080

for entry in entries:

    if "Immune" in entry :
        print("Immune" in entry)
        mask_Immune = cv2.imread('/home/maorvelous/Documents/Lab/deepLearning/MaskLou/'+ entry, cv2.IMREAD_GRAYSCALE)

        n_white_pix = np.sum(mask_Immune == 255)
        print('Number of white pixels:', n_white_pix)

        percentage = n_white_pix / ( total_of_pixels / 100 )
        print(" Percentage number of Pixels : ", percentage )

        ret, thresh = cv2.threshold(mask_Immune, 120, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        print(" Number of annotated Regions of Immune Cells: ", len(contours))
        mask_Tumor = cv2.imread('/home/maorvelous/Documents/Lab/deepLearning/MaskLou/' + entry.replace('Immune cells', 'Tumor'), cv2.IMREAD_GRAYSCALE)

        n_white_pix = np.sum(mask_Tumor == 255)
        print('Number of white pixels:', n_white_pix)

        percentage = n_white_pix / (total_of_pixels / 100)
        print(" Percentage number of Pixels : ", percentage)

        ret, thresh = cv2.threshold(mask_Tumor, 120, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        print(" Number of annotated Regions of Tumors: ", len(contours))

        mask_Tumor = cv2.imread(
            '/home/maorvelous/Documents/Lab/deepLearning/MaskLou/' + entry.replace('Immune cells', 'Other'),
            cv2.IMREAD_GRAYSCALE)

        stromaFileName = '/home/maorvelous/Documents/Lab/deepLearning/MaskLou/' + entry.replace('Immune cells', 'Stroma')
        if os.path.isfile(stromaFileName):
            mask_Tumor = cv2.imread(stromaFileName, cv2.IMREAD_GRAYSCALE)

            n_white_pix = np.sum(mask_Tumor == 255)
            print('Number of white pixels:', n_white_pix)

            percentage = n_white_pix / (total_of_pixels / 100)
            print(" Percentage number of Pixels : ", percentage)

            ret, thresh = cv2.threshold(mask_Tumor, 120, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

            print(" Number of annotated Regions of Others: ", len(contours))