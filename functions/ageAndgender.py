import cv2
import warnings
warnings.filterwarnings("ignore")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"



ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

def preditAgeGender(frame,bboxs):
    
    
        for (bbox) in bboxs:
            
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            
            
            crop = frame[y:y + h, x:x + w]
            if (crop.shape) > (300, 300):
                crop = cv2.resize(crop, (300, 300))
                
            
        
            # face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(crop, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            
            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]


            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]
            
                
            label="{},{}".format(gender,age)
            
            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2]+100, bbox[1]), (0,255,0),-1)       
            
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        
        return frame
