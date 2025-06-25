import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#load known faces
shano_image = face_recognition.load_image_file("known_faces/shano.jpg")
shano_encoding = face_recognition.face_encodings(shano_image)[0]

ma_image = face_recognition.load_image_file("known_faces/ma.jpg")
ma_encoding = face_recognition.face_encodings(ma_image)[0]

known_face_encodings = [shano_encoding, ma_encoding]
known_face_names = ["Shano", "Ma"]

#list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# get current time
now = datetime.now()
current_time = now.strftime("%Y:%m:%d")

f = open(f"{current_time}.csv", "w+", newline="")
lnwriter = csv.writer(f)