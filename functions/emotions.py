
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from keras.models import  load_model
from keras.utils import image_utils

# load model
model = load_model("C:/Face-Final/models/best_model.h5")

def predictEmotion(frame,bboxs):
    
    for (bbox) in bboxs:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
    
    
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image_utils.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index].capitalize()

        cv2.putText(frame, predicted_emotion, (bbox[0] + 20, bbox[1] - 100 ), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
        
        
      
    return frame