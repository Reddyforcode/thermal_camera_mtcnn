import cv2
from videocaptureasync import VideoCaptureAsync
from mtcnn_to_face_alignment import mtcnn_to_face_alignment


def init_cameras():
    #capture thermic frame
    cap_thermal = VideoCaptureAsync(thermal=True)
    cap_thermal.start()
    #capture normal frame
    cap_normal = VideoCaptureAsync(thermal=False)
    cap_normal.start()
    return cap_normal, cap_thermal

def stop_cameras(cap_normal, cap_thermal):
    cap_thermal.stop()
    cap_normal.stop()


cap_normal, cap_thermal = init_cameras()
face_detector = mtcnn_to_face_alignment()


while True:
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()
    _, thermal_frame = cap_thermal.read()

    _, normal_frame  = cap_normal.read()

    #thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_RGB2GRAY)
    thermal_frame = cv2.resize(thermal_frame, (704, 480))
    #thermal_frame = cv2.adaptiveThreshold(thermal_frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    normal_frame = normal_frame[31:429, 92:541]
    normal_frame = cv2.resize(normal_frame, (704, 480))
    cv2.circle(normal_frame, (92, 31), 2, (255, 255, 0))
    cv2.circle(normal_frame, (541, 429), 2, (255, 255, 0))
    

    try:
        #en vez de normal_frame mandar person box from tracker
        face_locations = face_detector.find_bboxes(normal_frame)
        print(face_locations)
        print(face_locations[0])
        for faces in face_locations:
            for face in faces:
                cv2.rectangle(thermal_frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
                cv2.rectangle(normal_frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
                #crop image thermal_frame and nroam
                #gender, age = cropped image
                #color = 
    except:
        pass
    cv2.imshow('thermal', thermal_frame)
    cv2.imshow('normal', normal_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
stop_cameras(cap_normal, cap_thermal)
cv2.destroyAllWindows()