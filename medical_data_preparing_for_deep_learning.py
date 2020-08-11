# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:38:13 2020

@author: serdarhelli
"""
import numpy as np
import os 
import numpy as np
from PIL import Image
x=np.load("C:/Users/sserd/Desktop/PROJE-DENTIST/x_train.npy")
def cross_validator(x,fold,w,h,c,filepath):
    import numpy as np
    r=len(x)%fold
    testshape=len(x)//fold
    test_value=len(x)//fold
    for i in range(0,fold-1):  
        split=len(x)-(test_value*i)
        split2=split-test_value
        x_test=np.copy(x[int(split2):int(split),:,:,:])
        x_test=np.reshape(x_test,(testshape,w,h,c))
        x_train1=np.copy(x[:int(split2),:,:,:])
        x_train2=np.copy(x[int(split):,:,:])
        x_train=np.concatenate((x_train1,x_train2),axis=None)
        x_train=np.reshape(x_train,(len(x)-testshape,w,h,c))
        np.save(filepath+"test"+np.str(i+1)+".npy",x_test)
        np.save(filepath+"train"+np.str(i+1)+".npy",x_train)
    testshaper=(len(x)//fold)+r
    split=len(x)-(test_value*(fold-1))
    split2=split-test_value-r
    x_test=np.copy(x[int(split2):int(split),:,:,:])
    x_train1=np.copy(x[:int(split2),:,:,:])
    x_train2=np.copy(x[int(split):,:,:])
    x_train=np.concatenate((x_train1,x_train2),axis=None)
    x_train=np.reshape(x_train,(len(x)-testshaper,w,h,c))
    x_test=np.reshape(x_test,(testshaper,w,h,c))
    np.save(filepath+"test"+np.str(fold)+".npy",x_test)
    np.save(filepath+"train"+np.str(fold)+".npy",x_train)
    return 


cross_validator(x, 11, 512, 512, 3, "C:/Users/sserd/Desktop/deneme/")

def medical_data_prepare(filepath,RGB,savepath):#all images must be same size
    import os 
    import numpy as np
    from PIL import Image
    dirs=sorted(os.listdir(filepath),key=len)
    img_x_train=Image.open(filepath+dirs[1])
    w,h=img_x_train.size
    img_x_train=np.asarray(img_x_train)
    img_x_train1=np.asarray(Image.open(filepath+dirs[2]))
    if RGB==False:
        canal=1
        x_train=np.concatenate((img_x_train[:,:,0],img_x_train1[:,:,0]),axis=None)
        for i in range (2,len(dirs)):
            img=np.asarray(Image.open(filepath+dirs[i]))
            if np.size(img.shape)==3:
                img=img[:,:,0]
                x_train=np.concatenate((x_train,img),axis=None)
            else:
                x_train=np.concatenate((x_train,img),axis=None)
    else :
        x_train=np.concatenate((img_x_train,img_x_train1),axis=None)
        canal=3
        for i in range (2,len(dirs)):
            img=np.asarray(Image.open(filepath+dirs[i]))
            x_train=np.concatenate((x_train,img),axis=None)
    x_train=np.reshape(x_train,(len(dirs),w,h,canal))
    np.save(savepath+"Data.npy",x_train)

medical_data_prepare("C:/Users/sserd/Desktop/7TP-DATASET-EDIT/",True,"C:/Users/sserd/Desktop/")


def medical_image_resize(filepath,size,savepath,extension):
    from PIL import Image
    import os 
    import numpy as np
    dirs=sorted(os.listdir(filepath),key=len)
    for i in range (0,len(dirs)):
        img=Image.open(filepath+dirs[i])
        (img.resize((size,size),Image.ANTIALIAS)).save(savepath+np.str(i+1)+"."+extension,extension, optimize=True)
   




medical_image_resize("C:/Users/sserd/Desktop/PROJE-DENTIST/7TP-Dataset-23072020/",512,"C:/Users/sserd/Desktop/deneme","jpeg")








