import cv2
import numpy as np
import glob
import os



for k in range(51):
    file_address="/home/airlab/PycharmProjects/pythonProject5/siamET-another/tools/image_result{}".format(k)
    path, dirs, files = next(os.walk(file_address))
    file_count = len(files)
    print(file_count)
    img_array=[]
    for i in range(file_count):
        file=file_address+'/{}.jpg'.format(i)
        img = cv2.imread(file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out=cv2.VideoWriter('project{}.avi'.format(k),cv2.VideoWriter_fourcc(*"DIVX"),30,size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()