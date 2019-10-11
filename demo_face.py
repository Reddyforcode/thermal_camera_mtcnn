import cv2
from videocaptureasync import *
from mtcnn_to_face_alignment import mtcnn_to_face_alignment

cap = VideoCaptureAsync()
cap.start()
face_detector = mtcnn_to_face_alignment()

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    try:
        face_locations = face_detector.find_bboxes(frame)
        print(face_locations)
        print(face_locations[0])
        for faces in face_locations:
            for face in faces:
                cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
    except:
        pass
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.stop()
cv2.destroyAllWindows()