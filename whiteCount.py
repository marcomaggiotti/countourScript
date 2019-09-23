import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


entries = os.listdir('/home/marco/Documents/Lab/deepLearning/MaskLou/')

for entry in entries[0:1]:
    #print(entry)
    mask = cv2.imread('/home/marco/Documents/Lab/deepLearning/MaskLou/'+ entry,
                     cv2.IMREAD_GRAYSCALE)
    n_white_pix = np.sum(mask == 255)
    #print('Number of white pixels:', n_white_pix)

    n_black_pix = np.sum(mask == 0)
    # print('Number of black pixels:', n_black_pix)

    ret, thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #print(contours)
    dataframes = []

    for contour_id, contour in enumerate(contours):
        for a in contour:
            print(contour)
            current_dataframe = pd.DataFrame(np.array(a), columns=['row', 'column'])
            current_dataframe['contour'] = contour_id
            dataframes.append(current_dataframe)

    contours_data = pd.concat(dataframes)
    contours_data.to_excel('filename.xlsx', sheet_name='Sheet1')

    df = pd.DataFrame({'Data': contours})
    writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    #cv2.imshow("Mask", mask)
    for contour in contours:
        print(contour)
        cv2.drawContours(mask, contour, -1, (0, 255, 0), 3)
