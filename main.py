import cv2
from functions.FastFaceNet import fastFaceBox
from functions.ageAndgender import preditAgeGender
from functions.emotions import predictEmotion
from functions.recognize import facerecognizer
import warnings
warnings.filterwarnings("ignore")


cap = cv2.VideoCapture(0)

padding=20
while True:
    ret, frame = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    
    
    frame,bboxs=fastFaceBox(frame)
    
    frame = preditAgeGender(frame,bboxs)
    
    frame = predictEmotion(frame,bboxs)
    
    frame = facerecognizer(frame,bboxs)
    

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows



# #  For Images


# frame = cv2.imread('test.jpg')
# padding=20

   
# gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame,bboxs=faceBox(faceNet,frame)


# for (bbox) in bboxs:
        
#     x = bbox[0]
#     y = bbox[1]
#     w = bbox[2]
#     h = bbox[3]
       
#     face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#     blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
#     genderNet.setInput(blob)
#     genderPred=genderNet.forward()
#     gender=genderList[genderPred[0].argmax()]


#     ageNet.setInput(blob)
#     agePred=ageNet.forward()
#     age=ageList[agePred[0].argmax()]
        
#     roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
#     roi_gray = cv2.resize(roi_gray, (224, 224))
#     img_pixels = image_utils.img_to_array(roi_gray)
#     img_pixels = np.expand_dims(img_pixels, axis=0)
#     img_pixels /= 255

#     predictions = model.predict(img_pixels)

#     # find max indexed array
#     max_index = np.argmax(predictions[0])

#     emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#     predicted_emotion = emotions[max_index]

#     label="{},{}".format(gender,age)
        
        
#     cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
#     cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
#     cv2.putText(frame, predicted_emotion, (bbox[0] + 20, bbox[1] - 50 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

 

# resized_img = cv2.resize(frame, (400, 600))
# cv2.imshow('Facial emotion analysis ', frame)


# cv2.waitKey(0)



