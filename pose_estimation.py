import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture(0)
pTime = 0

while True:
    isTrue, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # puts the dots on the landmarks and connects them
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm) # prints the landmarks along with id
            cx, cy = int(lm.x * w) , int(lm.y * h) #gives the exact pixel location of the x and y landmark
            cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED) #puts a circle over the cx and cy values

    cTime = time.time()
    fps = 1/(cTime - pTime)
    ptime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv.imshow('vid', img)
    cv.waitKey(1) 

