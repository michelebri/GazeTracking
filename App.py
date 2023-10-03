import mediapipe as mp
import cv2
import pyttsx3
import numpy as np
import time 
import tkinter as tk
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
print(socket)

socket.connect("tcp://localhost:5555")

def memorizza_nome():
    global nome_Video
    nome_Video = nome_entry.get()
    nome_label.config(text=f"Nome memorizzato: {nome_Video}")
    root.quit()

root = tk.Tk()
root.title("Inserisci Nome")

nome_entry = tk.Entry(root)
nome_entry.pack()

invio_button = tk.Button(root, text="Invio", command=memorizza_nome)
invio_button.pack()

nome_label = tk.Label(root, text="")
nome_label.pack()

root.mainloop()

print(f"Nome memorizzato: {nome_Video}")
parametro = 3
sguardi = {}
output_sguardo = cv2.VideoWriter(nome_Video + 'sguardo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))
output_plan_view = cv2.VideoWriter(nome_Video + 'plan.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))
output_frame = cv2.VideoWriter(nome_Video + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))
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
riprodotto_destra= False
riprodotto_sinistra= False
riprodotto_centro= False
while riproduci:
 
    socket.send_string("acquisisci")
    im_up = socket.recv()
    rgb_frame = cv2.imdecode(np.frombuffer(im_up, np.uint8), cv2.IMREAD_COLOR)         
    rgb_frame = cv2.flip(im,1)
    rgb_frame = cv2.resize(rgb_frame,(640,480))         
    ah = rgb_frame.copy()    
    h, w,c = rgb_frame.shape
    rgb_frame = cv2.flip(rgb_frame,1)
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

        cv2.imshow("2",rgb_frame)

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

        cv2.imshow("1",rgb_frame)



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
            cv2.imshow("aaa",ah)
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
            cv2.line(dst, nose_dst, (int(nose_dst[0] + (10*angolo_larghezza-(distanza_x*parametro))), nose_dst[1]+400), (0, 255, 0), 3) 
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
            if m_sguardo_frontale < m_sguardo_finale:
                theta = -theta

            if (400 - nose_dst[0] + (10*angolo_larghezza-(distanza_x*parametro))) > 0:
                theta = -theta
            print(theta)
            if theta > 10:
                key = "sinistra"
            elif theta < -10:
                key = "destra"
            else:
                key = "centro"
            cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

             if key in sguardi and not key == "no" :
                sguardi[key] += 1
            else:
                sguardi[key] = 0
            if tempo_trascorso > 0 and tempo_trascorso < 10:
                cv2.putText(rgb_frame, str( 10 - tempo_trascorso), (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 2)
                cv2.putText(rgb_frame, "Try to fix webcam in front of your face :)", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(rgb_frame, "the record will start, UP VOLUME OF SPEAKER TO MAX", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(rgb_frame, "Direct gaze in the direction you will hear", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if tempo_trascorso >10:

                cv2.putText(rgb_frame, "record: "+str(tempo_trascorso-10), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if tempo_trascorso >10 and tempo_trascorso < 18:
                if not riprodotto_destra:
                    socket.send_string("riproduci_GuardaFoglioADestra")
                    r = socket.recv()
                    riprodotto_destra = True 

                cv2.putText(rgb_frame, "right", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                sguardo_diretto = key;
                sguardo_obiettivo = "destra";
                if sguardo_diretto == "destra":
                    destra +=1
                if sguardo_diretto == "sinistra":
                    sinistra +=1
                if sguardo_diretto == "centro":
                    centro +=1
                destra_obiettivo +=1
         
            if tempo_trascorso >18 and tempo_trascorso < 26:
                if not riprodotto_sinistra:
                    socket.send_string("riproduci_GuardaPallaASinistra")

                    r = socket.recv()

                    riprodotto_sinistra = True 
                cv2.putText(rgb_frame, "left", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                sguardo_diretto = key;
                sguardo_obiettivo = "sinistra";
                if sguardo_diretto == "destra":
                    destra +=1
                if sguardo_diretto == "sinistra":
                    sinistra +=1
                if sguardo_diretto == "centro":
                    centro +=1
                sinistra_obiettivo +=1
 

                riprodotto_destra = False           
            if tempo_trascorso >26 and tempo_trascorso < 32:
                if not riprodotto_centro:
                    socket.send_string("riproduci_GuardamiAlCentro")

                    r = socket.recv()
                    riprodotto_centro= True
                cv2.putText(rgb_frame, "middle", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


                riprodotto_sinistra = False  
                sguardo_diretto = key;
                sguardo_obiettivo = "centro";
                if sguardo_diretto == "destra":
                    destra +=1
                if sguardo_diretto == "sinistra":
                    sinistra +=1
                if sguardo_diretto == "centro":
                    centro +=1
                centro_obiettivo +=1
            if tempo_trascorso >32 and tempo_trascorso < 36:
                if not riprodotto_sinistra:
                    socket.send_string("riproduci_GuardaPallaASinistra")

                    r = socket.recv()
                    riprodotto_sinistra= True

                cv2.putText(rgb_frame, "left", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                riprodotto_centro = False 
                sguardo_diretto = key;
                sguardo_obiettivo = "sinistra";
                if sguardo_diretto == "destra":
                    destra +=1
                if sguardo_diretto == "sinistra":
                    sinistra +=1
                if sguardo_diretto == "centro":
                    centro +=1
                sinistra_obiettivo +=1 

            if tempo_trascorso >36 and tempo_trascorso < 42:
                if not riprodotto_centro:
                    socket.send_string("riproduci_GuardamiAlCentro")

                    r = socket.recv()
                    riprodotto_centro = True 

                cv2.putText(rgb_frame, "middle", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                
                riprodotto_sinistra= False
                sguardo_diretto = key;
                sguardo_obiettivo = "centro";
                if sguardo_diretto == "destra":
                    destra +=1
                if sguardo_diretto == "sinistra":
                    sinistra +=1
                if sguardo_diretto == "centro":
                    centro +=1
                centro_obiettivo +=1 
            if tempo_trascorso >42 and tempo_trascorso < 48:
                if not riprodotto_destra:
                    socket.send_string("riproduci_GuardaFoglioADestra")

                    r = socket.recv()

                    riprodotto_destra= True

                cv2.putText(rgb_frame, "right", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(rgb_frame, key, (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                riprodotto_centro = False 
                riprodotto_destra= True
                sguardo_diretto = key;
                sguardo_obiettivo = "destra";
                if sguardo_diretto == "destra":
                    destra +=1
                if sguardo_diretto == "sinistra":
                    sinistra +=1
                if sguardo_diretto == "centro":
                    centro +=1
                destra_obiettivo +=1 
            cv2.imshow("Sguardo",rgb_frame)
            cv2.imshow("Plan view",dst)
            

            output_sguardo.write(rgb_frame)
            output_plan_view.write(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))

            if tempo_trascorso> 150:
                riproduci= False
    keycv = cv2.waitKey(50)
    if keycv == ord("q"):
        riproduci = False

with open(nome_Video+".txt", "w") as file:
    # Scrivi i dati nel file
    file.write("Obiettivi centro destra sinistra\n")
    file.write(str(centro_obiettivo) + "\n")
    file.write(str(destra_obiettivo) + "\n")
    file.write(str(sinistra_obiettivo) + "\n")
    file.write("Sguardi\n")
    file.write(str(centro) + "\n")
    file.write(str(destra) + "\n")
    file.write(str(sinistra) + "\n")

output_sguardo.release()
output_frame.release()
output_plan_view.release()