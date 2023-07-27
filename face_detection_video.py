import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture('test.mp4')

while video.isOpened():
    _,frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #1.1 is Scale Factor which means it specifies how much the image size is reduced at each image scale
    #minNeighbors is how many neighbors each candidate rectangle should have to retain it
    faces = face_cascade.detectMultiScale(gray,1.1,minNeighbors=4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),3)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()