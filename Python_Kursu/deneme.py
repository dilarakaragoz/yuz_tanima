import cv2, os
import numpy as np
from PIL import Image

tanıyıcı = cv2.face.LBPHFRecognizer_create()

dedektor = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    yuzornekleri=[]
    isimler = []
    for imagePaths in imagePaths:

        PIL_img = Image.open(imagePaths).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePaths)[-1].split(".")[1])
        print(id)

        yuzler = dedektor.detectMultiScale(img_numpy)

        for (x,y,w,h) in yuzler:

            yuzornekleri.append(img_numpy[y:y+h,x:x+w])
            isimler.append(id)

        return yuzornekleri, isimler

yuzler,isimler = getImagesAndLabels('veri')
tanıyıcı.train(yuzler, np.array(isimler))
tanıyıcı.save('deneme/deneme.yml')