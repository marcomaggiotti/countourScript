import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

'''
data= { 'ID' : [,,,,,,,,,,,],
        'Tumor': [ 250428, 19925, 20235, 187796, 279172, 165729, 215407, 76208, 46921, 148481, 229936, 7154],
        'Stroma' : [ 0, 29348, 2054, 67738, 15313, 17543, 53515, 1536, 30120, 12166, 24220, 27614],
        'Immune' : [110984, 74098, 27356, 37212, 63876, 19478,17776, 6501, 4560, 11614, 49521, 12557],
        'Others' : [ 8123, 3029, 0, 1092, 3401, 695, 0, 188, 14759, 2966, 5375, 0]},
'''

#annotation_dataframe= pd.DataFrame(data)
indexDataFrame = -1
maskDirPath = '/home/maorvelous/Documents/Lab/deepLearning/MaskLou/'

idSample = 000000
entries = os.listdir(maskDirPath)
total_of_pixels = 88363080
annotatedColor = 255
annotations_types = ['Immune cells', 'Tumor', 'Stroma', 'Others']

def countWhiteOnMask(entry, type):
    print(type in entry)

    mask_Immune = cv2.imread(maskDirPath + entry, cv2.IMREAD_GRAYSCALE)

    print(entry)
    n_white_pix = np.sum(mask_Immune == annotatedColor)
    print("Number of white pixels of ", type,":", n_white_pix)
    '''
    if type == "Immune cells":
        #print("dataFrame Immune cell ",annotation_dataframe.iloc[0][0][indexDataFrame] )
        averageCell = n_white_pix / annotation_dataframe.iloc[0][0][indexDataFrame]

    if type == "Tumor":
        #print("dataFrame Immune cell ", annotation_dataframe.iloc[0][1][indexDataFrame])
        averageCell = n_white_pix / annotation_dataframe.iloc[0][1][indexDataFrame]

    if type == "Stroma":
        averageCell = n_white_pix / annotation_dataframe.iloc[0][2][indexDataFrame]

    if type == "Others":
        averageCell = n_white_pix / annotation_dataframe.iloc[0][3][indexDataFrame]

    print( " Average white pixel for annotations ", type,": " , averageCell  )
    '''
    percentage = n_white_pix / (total_of_pixels / 100)
    print(" Percentage number of Pixels : ", percentage)

    ret, thresh = cv2.threshold(mask_Immune, 120, annotatedColor, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    print(" Number of annotated Regions of ",type, ": ", len(contours))

for entry in sorted(entries):

    if "Immune" in entry :

        indexDataFrame = indexDataFrame + 1
        print("Index of the data frame : ", indexDataFrame)
        id = entry.replace("Immune cells_mask.png","")
        countWhiteOnMask(entry, 'Immune cells')
        tumorFileName = entry.replace('Immune cells', 'Tumor')
        countWhiteOnMask(tumorFileName, 'Tumor')

        stromaFileName = entry.replace('Immune cells', 'Stroma')

        if os.path.isfile(maskDirPath + stromaFileName):
            countWhiteOnMask(stromaFileName, 'Stroma')

        othersFileName = entry.replace('Immune cells', 'Other')

        if os.path.isfile(maskDirPath + othersFileName):
            countWhiteOnMask(othersFileName, 'Others')