import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


entries = os.listdir('/home/marco/Documents/Lab/deepLearning/MaskLou/')
total_of_pixels = 88363080

for entry in entries:
    print(entry)

    print("Immune" in entry)
    #if "Immune" in entry :
    mask = cv2.imread('/home/marco/Documents/Lab/deepLearning/MaskLou/'+ entry,
                     cv2.IMREAD_GRAYSCALE)

    n_white_pix = np.sum(mask == 255)
    print('Number of white pixels:', n_white_pix)

    n_black_pix = np.sum(mask == 0)
    print('Number of black pixels:', n_black_pix)

    total_of_Pixels =  n_white_pix + n_black_pix
    percentage = n_white_pix / ( total_of_pixels / 100 )

    print(" Percentage number of Pixels : ", percentage )

    ret, thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    #print(contours)
    dataframes = []
    print(" Number of annotated Regions : ", len(contours))

    for contour_id, contour in enumerate(contours):
        #print(contour)
        current_dataframe = pd.DataFrame(index=range(1,10))
        #current_dataframe['contour'] = contour
        dataframes.append(current_dataframe)