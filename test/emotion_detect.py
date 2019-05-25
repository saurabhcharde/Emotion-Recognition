import cv2
import numpy as np
from keras.models import load_model

classifier = load_model('face_model.h5')
emotion_dict = {'0':'angry','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprise','6':'neutral'}

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture("ronaldo.mp4")
disp_emotion = None
skip = 0
while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face = small_frame[y:y+h,x:x+w ,:]
        face = np.array(face)
        face = np.resize(face, (48,48,3))
        face = np.reshape(face ,(1,48,48,3))
        face = face/255
        emotion = classifier.predict_classes(face)[0]
        if(skip%10==0):
            skip = 0
            disp_emotion = emotion_dict[str(emotion)]
        cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(small_frame, disp_emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', small_frame)
    skip+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
