import cv2
import numpy as np
import time
import os
import HandTrackingModule as ht
import Predictor as pt

########################
brushThickness = 10
drawColor = (255,0,255)
eraseColor = (0,0,0)
eraseThickness = 50
########################

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
prev_X = 0
prev_Y = 0
prev_Erase_X = 0
prev_Erase_Y = 0
count = 0
predicted_drawing = ""

detector = ht.HandTracker(min_detection_confidence=0.85)
predictor = pt.DoodlePredictor()
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img) # detect hands
    landmark_list = detector.findPosition(img, draw=False) # get coordinates of hand landmarks
    
    if len(landmark_list) != 0:
        # get coordinates of the tip of index and middle fingers
        index_X, index_Y = landmark_list[8][1:]
        middle_X, middle_Y = landmark_list[12][1:]

        fingers = detector.areFingersUp()
        
        # Detection Mode (Thumbs up)
        if fingers["Thumb"] and not fingers["Finger 1"] and not fingers["Finger 2"]:
            prev_X, prev_Y = 0,0
            prev_Erase_X, prev_Erase_Y = 0,0
            prediction = predictor.predict(img_canvas)
            if prediction:
                predicted_drawing = prediction

        # Erase Mode (index and middle fingers are up)
        elif fingers["Finger 1"] and fingers["Finger 2"] and not fingers["Finger 3"] and not fingers["Finger 4"] and not fingers["Thumb"]:
            prev_X, prev_Y = 0,0
            if prev_Erase_X == 0 and prev_Erase_Y == 0:
                prev_Erase_X, prev_Erase_Y = index_X, index_Y
            cv2.rectangle(img, (index_X,index_Y-15), (middle_X,middle_Y+15), drawColor, cv2.FILLED)

            # cv2.line(img, (prev_X,prev_Y), (index_X, index_Y), eraseColor, eraseThickness)
            cv2.line(img_canvas, (prev_Erase_X,prev_Erase_Y), (index_X, index_Y), eraseColor, eraseThickness)
        
        # Drawing Mode (index finger up, mdidle finger down)
        elif fingers["Finger 1"] and not fingers["Finger 2"] and not fingers["Finger 3"] and not fingers["Finger 4"] and not fingers["Thumb"]:
            cv2.circle(img, (index_X,index_Y), 15, drawColor, cv2.FILLED)
            prev_Erase_Y, prev_Erase_X = 0, 0
            if prev_X == 0 and prev_Y == 0:
                prev_X, prev_Y = index_X, index_Y

            cv2.line(img, (prev_X,prev_Y), (index_X, index_Y), drawColor, brushThickness)
            cv2.line(img_canvas, (prev_X,prev_Y), (index_X, index_Y), drawColor, brushThickness)
            prev_X, prev_Y = index_X, index_Y

        # Clear All (all fingers up)
        elif fingers["Finger 1"] and fingers["Finger 2"] and fingers["Finger 3"] and fingers["Finger 4"] and fingers["Thumb"]:
            img_canvas = np.zeros((720, 1280, 3), np.uint8)
            prev_X, prev_Y = 0,0
            prev_Erase_X, prev_Erase_Y = 0,0
            predicted_drawing = ""
            
        else:
            prev_X, prev_Y = 0,0
            prev_Erase_X, prev_Erase_Y = 0,0

    # Overlay canvas on top of webcam image
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,img_inv)
    img = cv2.bitwise_or(img,img_canvas)
    if predicted_drawing:
        cv2.putText(img, "You drew: " + predicted_drawing, (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 3)


    cv2.imshow("Image", img)
    cv2.imshow("Canvas", img_canvas)
    cv2.waitKey(1)
