# preprocess_input : adjust images for fitting to mobilnet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# loading model, as we are using model which was already made.
from tensorflow.keras.models import load_model
# multi-dimensional array and matrix data structures
import numpy as np
# for image, video process
import cv2

import matplotlib.pyplot as plt
import os

# configuration file, model
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

# keras model
model = load_model('models/mask_detector.model')


cap = cv2.VideoCapture(0)
ret, img = cap.read()

# https://overface.tistory.com/584
# defind how to compress file
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))


while cap.isOpened():

    ret, img = cap.read()
    # if it returns any frame, ret = 1
    # else ret = 0, so break.
    if not ret:
        break

    # output order : height, width, channel
    h, w = img.shape[:2]

    # preprocessing for face detection
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))

    # input data for facenet(model)
    facenet.setInput(blob)


    detections = facenet.forward()

    result_img = img.copy()

    # Detect Faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)


        face = img[y1:y2, x1:x2]

        face_input = cv2.resize(face, dsize=(224, 224))
        #BGR to RGB
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)
        
        mask, nomask = model.predict(face_input).squeeze()


        if mask > nomask:
            # Green
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            #
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)

        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

    out.write(result_img)
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
