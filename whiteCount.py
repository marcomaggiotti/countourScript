import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


entries = os.listdir('/home/marco/Documents/Lab/deepLearning/MaskLou/')
dataframes = []
for entry in entries[0:4]:
    print(entry)
    mask = cv2.imread('/home/marco/Documents/Lab/deepLearning/MaskLou/'+ entry,
                     cv2.IMREAD_GRAYSCALE)
    n_white_pix = np.sum(mask == 255)
    #print('Number of white pixels:', n_white_pix)

    n_black_pix = np.sum(mask == 0)
    # print('Number of black pixels:', n_black_pix)

    ret, thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #print(contours)


    for contour in contours:
        for a,b in enumerate(contour):
            #print(contour)
            current_dataframe = pd.DataFrame(b, columns=['row', 'column'])
            current_dataframe['contour'] = a
            current_dataframe['filename'] = entry
            current_dataframe['size'] = len(contour)
            current_dataframe['White Number'] = n_white_pix
            dataframes.append(current_dataframe)

    contours_data = pd.concat(dataframes)
contours_data.to_excel('filename.xlsx', sheet_name='Sheet1')
#cv2.imshow("Mask", mask)
for contour in contours:
    #print(contour)
    cv2.drawContours(mask, contour, -1, (0, 255, 0), 3)
