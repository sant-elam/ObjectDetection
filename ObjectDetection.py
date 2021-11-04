# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:41:51 2021

@author: Santosh
"""

import cv2 as cv
import mediapipe as mp

objectron = mp.solutions.objectron
drawing_utils = mp.solutions.drawing_utils

drwLandmark = drawing_utils.DrawingSpec((200, 50, 50), thickness=3, circle_radius=5)
drwConnections = drawing_utils.DrawingSpec((50, 200, 50), thickness=3)


cap = cv.VideoCapture(0)

model = objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.3,
                            model_name="Cup")

while True:
    
    result, image_org = cap.read()
    
    height,width = image_org.shape[0:2]
    if result:
        image = cv.cvtColor(image_org, cv.COLOR_BGR2RGB)
        
        output = model.process(image)
        
        if output.detected_objects:
            
            for objects in output.detected_objects:               
                
                #print(objects.rotation)
                drawing_utils.draw_landmarks(image = image_org, 
                                             landmark_list = objects.landmarks_2d,
                                             connections = objectron.BOX_CONNECTIONS,
                                             landmark_drawing_spec= drwLandmark,
                                             connection_drawing_spec=drwConnections)
                
                drawing_utils.draw_axis(image_org, objects.rotation, objects.translation)
                
                '''
                obj = objects.landmarks_2d.landmark[0]
                x = int (obj.x * width)
                y = int (obj.y * height)
                
                cv.circle(image_org, (x,y), 6, (0, 0, 255) )
                
                
                for obj in objects.landmarks_2d.landmark[1:3]:
                    x = int (obj.x * width)
                    y = int (obj.y * height)
                    
                    cv.circle(image_org, (x,y), 6, (0, 255, 255) )
                
                '''
                
                
                
                    
                    
        
    cv.imshow("Objectron", image_org)
    if cv.waitKey(30) & 255 == 27:
        break
    
cv.destroyAllWindows()
cap.release()