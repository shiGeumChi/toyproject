# 이미지 오픈소스 라이브러리
# pip install opencv-python
import cv2

# dlib is a toolkit for making real world machine learning and data analysis applications in C++
import dlib

# A series of convenience functions to make basic image processing
# functions such as translation, rotation, resizing, skeletonization,
# and displaying Matplotlib images easier with OpenCV and both Python 2.7 and Python 3.
from imutils import face_utils, resize

# 대규모 다차원 배열을 쉽게 처리 할 수 있도록 지원하는 파이썬의 라이브러리
import numpy as np

# 리턴 타입 numpy.ndarray
orange_img = cv2.imread('orange.jpg')
orange_img = cv2.resize(orange_img, dsize=(512, 512))

# Returns the default face detector
detector = dlib.get_frontal_face_detector()

# This object is a tool that takes in an
# image region containing some object and
# outputs a set of point locations that define
# the pose of the object. The classic example
# of this is human face pose prediction, where
# you take an image of a human face as input and
# are expected to identify the locations of important
# facial landmarks such as the corners of the mouth and eyes,
# tip of the nose, and so forth.
predictor = dlib.shape_predictor('C:/Users/shiGeumChi/ToyProject/annoyingOrange/shape_predictor_68_face_landmarks.dat')

# https://opencv-tutorial.readthedocs.io/en/latest/intro/intro.html#capture-live-video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # frame by frame
    ret, img = cap.read()

    # if frame doesn't fit?
    if not ret:
        break

    # faces의 데이터 타입?
    faces = detector(img)
    # 리턴 타입 numpy.ndarray
    result = orange_img.copy()

    # faces가 있을 때. 즉, detector가 제대로 작동했을때.
    if len(faces) > 0:
        # faces의 첫단을 face에 저장.
        # faces[0]이 무엇? (좌표값을 불러올수 있는 객체)
        face = faces[0]

        # 얼굴 최대크기를 아우르는 좌표값 저장.
        # x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # 해당 크기만큼 이미지 crop하여 저장.
        # face_img = img[y1:y2, x1:x2].copy()

        # 전체 이미지, 얼굴 사진을 input -> 찍힌 점대로 shape 파악.
        shape = predictor(img, face)
        # shape값을 ndarry로 저장.
        shape = face_utils.shape_to_np(shape)

        # for p in shape:
        # cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        # eyes 값 저장.
        le_x1 = shape[36, 0]
        le_y1 = shape[37, 1]
        le_x2 = shape[39, 0]
        le_y2 = shape[41, 1]

        le_margin = int((le_x2 - le_x1) * 0.18)

        re_x1 = shape[42, 0]
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0]
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.18)

        left_eye_img = img[le_y1 - le_margin:le_y2 + le_margin, le_x1 - le_margin:le_x2 + le_margin].copy()
        right_eye_img = img[re_y1 - re_margin:re_y2 + re_margin, re_x1 - re_margin:re_x2 + re_margin].copy()

        # 눈 crop 한 것 resize
        left_eye_img = resize(left_eye_img, width=100)
        right_eye_img = resize(right_eye_img, width=100)

        #
        result = cv2.seamlessClone(
            left_eye_img,
            # 배경
            result,
            # np.full ???
            # 왜 :2 부터일까
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            # 오렌지에 들어갈 왼쪽 눈 위치
            (100, 160),
            cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone(
            right_eye_img,
            # 배경
            result,
            # 왜 :2 부터일까
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            # 위치
            (250, 160),
            cv2.MIXED_CLONE
        )

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_img = img[mouth_y1 - mouth_margin:mouth_y2 + mouth_margin,
                    mouth_x1 - mouth_margin:mouth_x2 + mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (180, 330),
            cv2.MIXED_CLONE
        )

        # cv2.imshow('left', left_eye_img)
        # cv2.imshow('right', right_eye_img)
        # cv2.imshow('mouth', mouth_img)
        # cv2.imshow('face', face_img)

        cv2.imshow('result', result)

    # cv2.imshow('img', img)ㅂ
    if cv2.waitKey(1) == ord('q'):
        break

    # 0xFF (16진수) = 255 (10진수) = 11111111(2진수)
    # 1ms 대기
    # ord 유니코드 리턴 (char와 반대)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break