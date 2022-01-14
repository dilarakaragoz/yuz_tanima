import cv2
import numpy as np

taniyici = cv2.face.LBHFaceRecognizer_create()
taniyici.read('deneme/deneme.yml')
yolsiniflandirici = "haarcascade_frontalface_default.xml"

yuzsiniflandirici = cv2.CascadeClassifier(yolsiniflandirici);
font = cv2.VideoCapture(1)
vid_cam = cv2.VideoCapture(0)
while True:
    ret, kamera = vid_cam.read()
    gri = cv2.cvtColor(kamera,cv2.COLOR_BAYER_BG2GRAY)
    yuzler = yuzsiniflandirici.detectMultiScale(gri,1.2,5)

    for(x,y,w,h) in yuzler:

        cv2.rectangle(kamera, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        Id,conf = taniyici.predict(gri[y:y+h,x:x+w])
        print(Id)

        if(Id == 1):
            Id = "isim1"
        elif(Id == 2):
            Id = "isim2"
        elif(Id == 3):
            Id = "isim3"

        cv2.cv2.rectangle(kamera, (x-22,y-90), (x+w+22,y-22), (0,255,0), -1)

        cv2.putText(kamera, str(Id), (x,y-40), font, 2, (255,255,255), 3)

    cv2.imshow('kamera',kamera)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vid_cam.release()
cv2.destroyAllWindows()