# face_utils.py
import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, authorized_dir="authorized_face"):
        self.authorized_faces = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.load_authorized_faces(authorized_dir)

    def load_authorized_faces(self, directory="authorized_face"):
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            img = cv2.imread(path)
            if img is not None:
                face = self.extract_face(img)
                if face is not None:
                    self.authorized_faces.append(face)
                    print("Authorized face loaded:", filename)

    def extract_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            return cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        return None

    def is_authorized(self, frame, bbox):
        # Cắt vùng bounding box người (YOLO)
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]

    # Dò khuôn mặt trong vùng người
        face = self.extract_face(person_crop)
        if face is None:
            return False

    # So sánh với từng khuôn mặt được cấp phép
        for auth_face in self.authorized_faces:
            diff = cv2.absdiff(auth_face, face)
            score = np.sum(diff)
            if score < 20000000:
                return True
        return False
