import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#load known faces
shano_image = face_recognition.load_image_file("faces/shano.jpg")
shano_encoding = face_recognition.face_encodings(shano_image)[0]

sanvi_image = face_recognition.load_image_file("faces/sanvi.jpg")
sanvi_encoding = face_recognition.face_encodings(sanvi_image)[0]

known_face_encodings = [shano_encoding, sanvi_encoding]
known_face_names = ["Shano", "sanvi"]

#list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# get current time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]

        # Add the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present ", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
            
            if name in students:
                students.remove(name)
                now = datetime.now()
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name + " is present at ", current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
