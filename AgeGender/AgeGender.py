import cv2
import numpy as np

age_model = 'age_net.caffemodel'
age_proto = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'

age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#image to predict
image = cv2.imread('images/1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#scaleFactor: samller scale means more scales to be detectedb (slower but more precise)
faces_large = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))

faces_small = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

all_faces = list(faces_large)
for (x, y, w, h) in faces_small:
    is_Unique = True
    for (x2, y2, w2, h2) in faces_large:
        if abs(x-x2) < w2 / 2 and abs(y-y2) < h2 / 2:
            is_Unique = False
            break
    if is_Unique: 
        all_faces.append((x, y, w, h))
    
for (x, y, w, h) in all_faces:
    face_img = image[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    gender_confidence = gender_preds[0].max()

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    age_confidence = age_preds[0].max()
    
    if gender_confidence > 0.75 and age_confidence > 0.75:
        label = f"{gender}, {age}"
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow('Age and Gender Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()