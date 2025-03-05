import cv2
import mediapipe as mp
import math

zoom = False
smoothZoom = 1
smoothZoomAlpha = 0.1
scalefactor = 0.01

#Necessary models for hand recognization
mpHands = mp.solutions.hands.Hands(max_num_hands = 2, min_detection_confidence = 0.9, min_tracking_confidence = 0.7)

#Necessary model for drawing stuff
mpDraw = mp.solutions.drawing_utils

landmarkStyleRight = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)  # Blue landmarks
connectionStyleRight = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)  # White connections


#Starting the video capture
cap = cv2.VideoCapture(0)

#error if webcam cant be accessed
if not cap.isOpened():
    raise IOError("Webcam cannot be accessed")

#Start of the video loop
while True:
    ret,frame = cap.read()
    leftHand = None
    rightHand = None
    #Breaking the loop if frame cannot be gathered
    if not ret:
        break

    frame = cv2.flip(frame,1)

    #converting the BGR format of the frames to RGB
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #processing the frame to find the hand landmark
    results = mpHands.process(rgbFrame)

    if results.multi_hand_landmarks and len(results.multi_handedness) == 2:

        for idx, landmarks in enumerate(results.multi_hand_landmarks):

            label = results.multi_handedness[idx].classification[0].label

            if label == "Left":
                leftHand = landmarks
            elif label == "Right":
                rightHand = landmarks
            
            if rightHand:
                mpDraw.draw_landmarks(frame, rightHand, mp.solutions.hands.HAND_CONNECTIONS, landmarkStyleRight, connectionStyleRight)

                if zoom:

                    #tracking the thumb and index finger
                    thumbTip = rightHand.landmark[4]
                    indexTip = rightHand.landmark[8]

                    t1, t2, t3 = int(thumbTip.x * frame.shape[1]), int(thumbTip.y * frame.shape[0]), int(thumbTip.z)
                    i1, i2, i3 = int(indexTip.x * frame.shape[1]), int(indexTip.y * frame.shape[0]), int(indexTip.z)

                    distance =  math.sqrt((t2 - i2)**2 + (t1 - i1)**2 +  (t3 - i3)**2)
                    midPoint = (int((t1 + i1)/2), int((t2+i2)/2))
                    
                    cv2.line(frame, (i1,i2), (t1,t2), (255,0,0), 1)
                    cv2.putText(frame, f"{distance:.2f}" ,  midPoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1 )
                    
                    scalefactor = 0.01
                    scalefactor = max(0.01,min(0.2, scalefactor * (1 + (distance-20) * 0.1 )))                    

            if leftHand:
                mpDraw.draw_landmarks(frame, leftHand, mp.solutions.hands.HAND_CONNECTIONS)

                if zoom:
                    #tracking the thumb and index finger
                    thumbTip = leftHand.landmark[4]
                    indexTip = leftHand.landmark[8]

                    t1, t2, t3 = int(thumbTip.x * frame.shape[1]), int(thumbTip.y * frame.shape[0]), int(thumbTip.z)
                    i1, i2, i3 = int(indexTip.x * frame.shape[1]), int(indexTip.y * frame.shape[0]), int(indexTip.z)

                    distance =  math.sqrt((t2 - i2)**2 + (t1 - i1)**2 +  (t3 - i3)**2)
                    midPoint = (int((t1 + i1)/2), int((t2+i2)/2))
                    
                    cv2.line(frame, (i1,i2), (t1,t2), (0,0,255), 1)
                    cv2.putText(frame, f"{distance:.2f}" ,  midPoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1 )

                    zoomfactor = 1 + scalefactor * (distance - 30)
                    zoomfactor = max(1, min(15, zoomfactor))

                    smoothZoom = (1-smoothZoomAlpha) * smoothZoom + smoothZoomAlpha * zoomfactor

                    height, width, _ = frame.shape

                    if zoomfactor > 1:
                        centerX, centerY = midPoint
                    else:
                        centerX, centerY = int(width/2), int(height/2)
                    newHeight = (height/smoothZoom)
                    newWidth = (width/smoothZoom)

                    x1 = int(max( centerX - newWidth//2 , 0))
                    y1 = int(max( centerY - newHeight//2 , 0))
                    x2 = int(min( centerX + newWidth//2 , width - 2))
                    y2 = int(min( centerY + newHeight//2 , height - 2))

                    croppedFrame = frame[y1:y2, x1:x2]
                    frame = cv2.resize(croppedFrame, (width,height), interpolation=cv2.INTER_LINEAR)                

    elif results.multi_hand_landmarks: #checking if there are any hands in the frame
        #Iterating through each detected hand's landmark (landmarks are the difeerent parts of a hand)
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            if zoom:
                #tracking the thumb and index finger
                thumbTip = landmarks.landmark[4]
                indexTip = landmarks.landmark[8]

                t1, t2, t3 = int(thumbTip.x * frame.shape[1]), int(thumbTip.y * frame.shape[0]), int(thumbTip.z)
                i1, i2, i3 = int(indexTip.x * frame.shape[1]), int(indexTip.y * frame.shape[0]), int(indexTip.z)

                distance =  math.sqrt((t2 - i2)**2 + (t1 - i1)**2 + + (t3 - i3)**2)
                midPoint = (int((t1 + i1)/2), int((t2+i2)/2))
                
                cv2.line(frame, (i1,i2), (t1,t2), (0,0,255), 3)
                cv2.putText(frame, f"{distance:.2f}" ,  midPoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2 )

                scalefactor = 0.01
                zoomfactor = 1 + scalefactor * (distance - 30)
                zoomfactor = max(1, min(3, zoomfactor))

                smoothZoom = (1-smoothZoomAlpha) * smoothZoom + smoothZoomAlpha * zoomfactor

                height, width, _ = frame.shape

                if zoomfactor > 1:
                    centerX, centerY = midPoint
                else:
                    centerX, centerY = int(width/2), int(height/2)
                newHeight = (height/smoothZoom)
                newWidth = (width/smoothZoom)

                x1 = int(max( centerX - newWidth//2 , 0))
                y1 = int(max( centerY - newHeight//2 , 0))
                x2 = int(min( centerX + newWidth//2 , width - 2))
                y2 = int(min( centerY + newHeight//2 , height - 2))

                croppedFrame = frame[y1:y2, x1:x2]
                frame = cv2.resize(croppedFrame, (width,height), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Video Window", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        zoom = not zoom

cap.release()
cv2.destroyAllWindows()