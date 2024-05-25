import cv2
import pytesseract
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("C:\\Computer_Vision\\License_plate_recognition\\serviceAccountKey.json")
firebase = firebase_admin.initialize_app(cred,{
    'databaseURL':"https://number-plate-recognition-73ae6-default-rtdb.firebaseio.com/",
    'storageBucket':"number-plate-recognition-73ae6.appspot.com"
})
ref = db.reference("/")

folderPath = 'C:\\Computer_Vision\\License_plate_recognition\\images'

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

platecascade = cv2.CascadeClassifier(
    "C:\\Computer_Vision\\License_plate_recognition\\haarcascade_russian_plate_number.xml")
minArea = 500
cap = cv2.VideoCapture(0)
count = 0
# imgBackground = cv2.imread('C:\\Computer_Vision\\License_plate_recognition\\background.png')
# imgMode = cv2.imread('C:\\Computer_Vision\\License_plate_recognition\\Rectangle 4.png')

while True:
    ret, frame = cap.read()
    # imgBackground[169:169+720,76:76+1280]=frame
    # imgBackground[35:35+720,842:842+434]=imgMode
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    numberplates = platecascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in numberplates:
        area = w*h
        if area > minArea:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Number Plate Detected", (x, y-5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            frameReg = frame[y:y+h, x:x+w]

            kernel = np.ones((1, 1), np.uint8)
            frameReg = cv2.dilate(frameReg, kernel, iterations=1)
            frameReg = cv2.erode(frameReg, kernel, iterations=1)
            plate_gray = cv2.cvtColor(frameReg, cv2.COLOR_BGR2GRAY)
            (thresh, frameReg) = cv2.threshold(
            plate_gray, 127, 255, cv2.THRESH_BINARY)

        cv2.imshow('REG', frameReg)
    cv2.imshow("Result", frame)
    


    if(cv2.waitKey(1) == ord('s')):
        imgname = f"IMAGE{count+1}.jpg"
        cv2.imwrite(f"{folderPath}/{imgname}", frameReg)
        cv2.rectangle(frame, (0, 200), (640, 300), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, "SCAN SAVED", (15, 265),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(500)
        count += 1

        read = pytesseract.image_to_string(frameReg)
        read = ''.join(e for e in read if e.isalnum())
        print(read)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp)
        data = {"PlateNumber":read,"Time":timestamp}
        ref.child("Cars").push(data)
        
        fileName = f'{folderPath}/{imgname}'
        print(fileName)
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)
        
        
        
    
        
