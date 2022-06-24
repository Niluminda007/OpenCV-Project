import cv2
import mediapipe as mp
import time

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils



def fastFaceBox(frame):
    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

        

        

        start = time.time()


        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results = face_detection.process(frame)
            
        # Convert the image color back so it can be displayed
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

        width = frame.shape[1]
        height = frame.shape[0]
        
        
            
        bboxs=[]
            
        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(frame, detection)
                print(id, detection)

                bBox = detection.location_data.relative_bounding_box

                x = int(bBox.xmin * width)
                    
                y = int(bBox.ymin * height)
                    
                w = int(bBox.width * width)
                    
                h = int(bBox.height * height)
                bboxs.append([x,y,w,h])
                    
                    
                h, w, c = frame.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                

                    




        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(frame, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)



    return frame, bboxs