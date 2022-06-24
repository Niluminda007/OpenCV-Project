from tkinter import Frame
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np







model = load_model('C:/Face-Final/models/keras_model.h5' , compile=False)


def get_className(classNo):
    if classNo==0:
        return "Ushan"
    elif classNo==1:
        return "Tony"
    elif classNo==2:
        return "Kugan"
    elif classNo==3:
        return "Janis"
    elif classNo==4:
        return "Leo"
    else:
        return " "



def facerecognizer(frame,bboxs):
    
    for (x,y,w,h) in bboxs:
        crop_img=frame[y:y+h,x:x+h]
        img=cv2.resize(crop_img, (224,224))
        img=img.reshape(1, 224, 224, 3)
        prediction=model.predict(img)
        classIndex=np.argmax(prediction,axis =1)
        
        cv2.putText(frame, str(get_className(classIndex)),(x + w -20, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0),1, cv2.LINE_AA)
    return frame
        
        
  