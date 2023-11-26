import cv2
import mediapipe as mp
import time


class HandTracker():
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.detection_conf = min_detection_confidence
        self.tracking_conf = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_conf,
                                        min_tracking_confidence=self.tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils # for drawing hand landmarks

        self.fingertips_ids = [4, 8, 12, 16, 20]

    # detect and draw hands
    def findHands(self, img, drawHands=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # have to convert img to RGB for mediapipe
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks: # for each hand

                if drawHands:
                    self.mp_draw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS) # draw connections on hand

        return img
    
    # find the position of each hand landmark
    def findPosition(self, img, handNum=0, draw=True):
        self.landmark_lists = []

        if self.results.multi_hand_landmarks:
            currHand = self.results.multi_hand_landmarks[handNum]

            for index, landmark in enumerate(currHand.landmark):
                h, w, c = img.shape
                x_coor, y_coor = int(landmark.x*w), int(landmark.y*h)
                # print(index, x_coor, y_coor)
                self.landmark_lists.append([index, x_coor, y_coor])
                if draw:
                    cv2.circle(img, (x_coor, y_coor), 20, (0, 0, 255), cv2.FILLED)


        return self.landmark_lists
    
    def areFingersUp(self):
        fingers = {}

        # Thumb
        if self.landmark_lists[self.fingertips_ids[0]][1] > self.landmark_lists[self.fingertips_ids[0] - 1][1]:
            fingers["Thumb"] = True
        else:
            fingers["Thumb"] = False

        # Fingers
        for id in range(1, 5):
            if self.landmark_lists[self.fingertips_ids[id]][2] < self.landmark_lists[self.fingertips_ids[id] - 2][2]:
                fingers["Finger " + str(id)] = True
            else:
                fingers["Finger " + str(id)] = False

        return fingers

def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0

    detector = HandTracker()

    while True:
        success, img = cap.read() # read from webcam

        cTime = time.time()
        FPS = int(1/(cTime - pTime)) # calculate FPS
        pTime = cTime

        cv2.putText(img, "FPS: " + str(FPS), (10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0), 3) # display FPS


        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()