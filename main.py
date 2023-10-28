import numpy as np
import cv2

car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture('car_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nesne tespiti icin
    roi = gray[100:300, 200:800]
    #height and width

    cars = car_classifier.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in cars:
        x, y, w, h = x + 200, y + 100, w, h  # ROI koordinatlarını orijinal koordinatlara çevirin
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
