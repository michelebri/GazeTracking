import mediapipe as mp
import cv2
import pyttsx3
import numpy as np
import time 
import tkinter as tk

import zmq

parametro = 5
sguardi = {}
mp_face_mesh = mp.solutions.face_mesh


LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
riproduci = True
inizio = time.time()
destra = 0
centro = 0
sinistra = 0
destra_obiettivo = 0
centro_obiettivo = 0
sinistra_obiettivo = 0
cap = cv2.VideoCapture(0)
riprodotto_destra= False
riprodotto_sinistra= False
riprodotto_centro= False
while riproduci:
    tempo_trascorso = (int(time.time()-inizio))
 
    ret,frame = cap.read()

    rgb_frame = cv2.resize(frame,(640,480))  
    h, w,c = rgb_frame.shape
    print(h)

    rgb_frame = cv2.flip(rgb_frame,1)
    ah = rgb_frame.copy()    

    dst = cv2.imread("dest_2d.jpg")
    distanza_x = 0
    face_3d = []
    face_2d = []


    results = face_mesh.process(rgb_frame)

    def draw_eye_line(frame, center, average_point):
        cv2.line(frame, tuple(center), tuple(average_point), (0, 0, 255), 2)
    

    if results.multi_face_landmarks:

        tempo_trascorso = int(time.time() -inizio)
        mesh_points = np.array(
            [np.multiply([p.x, p.y], [w, h]).astype(int) for p in results.multi_face_landmarks[0].landmark]
        )

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        centro_sinistra = np.array([l_cx, l_cy], dtype=np.int32)
        centro_destra = np.array([r_cx, r_cy], dtype=np.int32)
        cv2.circle(rgb_frame, centro_sinistra, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(rgb_frame, centro_destra, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

        punti_occhio_sinistro = mesh_points[LEFT_EYE]
        punti_occhio_destro = mesh_points[RIGHT_EYE]

        bordi_sinistro = tuple(punti_occhio_sinistro[punti_occhio_sinistro[:, 0].argmin()]) + tuple(
            punti_occhio_sinistro[punti_occhio_sinistro[:, 0].argmax()]
        )
        media_occhio_sinistro = (
            (bordi_sinistro[0] + bordi_sinistro[2]) // 2,
            (bordi_sinistro[1] + bordi_sinistro[3]) // 2,
        )

        bordi_destro = tuple(punti_occhio_destro[punti_occhio_destro[:, 0].argmin()]) + tuple(
            punti_occhio_destro[punti_occhio_destro[:, 0].argmax()]
        )
        media_occhio_destro = (
            (bordi_destro[0] + bordi_destro[2]) // 2,
            (bordi_destro[1] + bordi_destro[3]) // 2,
        )

        draw_eye_line(rgb_frame, centro_destra, media_occhio_destro)
        draw_eye_line(rgb_frame, centro_sinistra, media_occhio_sinistro)




        distanza_x = (media_occhio_destro[0] - centro_destra[0]) + (media_occhio_sinistro[0] - centro_sinistra[0]) 
        distanza_y = media_occhio_destro[1] - centro_destra[1] + (media_occhio_sinistro[1] - centro_sinistra[1])  # da aggiungere

        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx ==291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x* w, lm.y*h)
                        nose_3d = (lm.x*w,lm.y*h,lm.z *8000)
                    x,y = int(lm.x*w), int(lm.y*h)
                    cv2.circle(ah,[x,y],1,(255,0,255),15,cv2.LINE_AA)
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])
            face_2d = np.array(face_2d, dtype= np.float64)
            face_3d = np.array(face_3d, dtype = np.float64)
            
            focal_length = 1*w
            
            cam_matrix = np.array([[focal_length, 0 , h/2],
                                    [0,focal_length, w/2],
                                    [0,0,1]])

            dist_matrix = np.zeros((4,1),dtype=np.float64)
            _,rot_vec,trans_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)

            rmat,jac = cv2.Rodrigues(rot_vec)
            angoli,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)
                    
            angolo_lunghezza = angoli[0]*360
            angolo_larghezza = angoli[1]*360
            z = angoli[2]*360

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            if abs(distanza_x) > 2:
                p2 = (int(nose_2d[0]+angolo_larghezza - distanza_x *parametro), int(nose_2d[1]-angolo_lunghezza))
            else:

                p2 = (int(nose_2d[0]+angolo_larghezza), int(nose_2d[1] -angolo_lunghezza))

            cv2.line(rgb_frame,p1,p2,(255,0,0),3)



            nose_dst = [315,104] 
            cv2.line(dst, nose_dst, (nose_dst[0], nose_dst[1]+400), (0, 255, 0), 3) 
            cv2.line(dst, nose_dst, (int(nose_dst[0] + (20*angolo_larghezza-(distanza_x*parametro))), nose_dst[1]+400), (255, 0, 0), 3) 
            eps = 0.01
            x1_sguardo_frontale, y1_sguardo_frontale = nose_dst
            x2_sguardo_frontale, y2_sguardo_frontale = nose_dst[0], nose_dst[1] + 400

            x1_sguardo_finale, y1_sguardo_finale = nose_dst
            x2_sguardo_finale = int(nose_dst[0] + (10 * angolo_larghezza - (distanza_x * parametro)))
            y2_sguardo_finale = nose_dst[1] + 400

            m_sguardo_frontale = (y2_sguardo_frontale - y1_sguardo_frontale) / (x2_sguardo_frontale - x1_sguardo_frontale+eps)
            m_sguardo_finale = (y2_sguardo_finale - y1_sguardo_finale) / (x2_sguardo_finale - x1_sguardo_finale+eps)

            angolo_rad = np.arctan(abs((m_sguardo_finale - m_sguardo_frontale) / (1 + m_sguardo_frontale * m_sguardo_finale)))
            theta = np.degrees(angolo_rad)
            if (400 - nose_dst[0] + (10*angolo_larghezza-(distanza_x*parametro))) > 0:
                theta = -theta
            if theta > 7:
                key = "sinistra"
            elif theta < -7:
                key = "destra"
            else:
                key = "centro"
            cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sguardo",rgb_frame)
    cv2.imshow("Plan view",dst)
    keycv = cv2.waitKey(50)
    if keycv == ord("q"):
        riproduci = False


