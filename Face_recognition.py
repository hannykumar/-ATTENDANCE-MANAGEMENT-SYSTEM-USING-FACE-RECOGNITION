import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

path = 'images'
images = []
classNames = [] 
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

encoded_face_train = findEncodings(images)
if not os.path.isfile('Attendance.csv'):
    with open('Attendance.csv', 'w') as f:
        pass  
last_entry_times = {}

def markAttendance(name):
    global last_entry_times
    
    now = datetime.now()
    current_time = now.strftime('%I:%M:%S:%p')
    current_date = now.strftime('%d-%B-%Y')
    
    if name not in last_entry_times:
        last_entry_times[name] = now
        write_attendance(name, current_time, current_date)
    else:
        time_diff = now - last_entry_times[name]
        if time_diff.total_seconds() >= 4 * 60 * 60:  
            last_entry_times[name] = now
            write_attendance(name, current_time, current_date)

def write_attendance(name, time, date):
    with open('Attendance.csv','a') as f:
        f.writelines(f'{name}, {time}, {date}\n')
cap = cv2.VideoCapture(0)  
while True:
    success, img = cap.read()
    if not success:
        print("Failed to get frame from the webcam.")
        break

    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame) 


   #DV

    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            markAttendance(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            
            known_distances = face_recognition.face_distance(encoded_face_train, encode_face)
            min_distance = min(known_distances)
            
            if min_distance < 0.5:  
                name = classNames[np.argmin(known_distances)].upper().lower()
                markAttendance(name)
            else:
                name = "Unknown"
                
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
