import zmq
import cv2
from naoqi import ALProxy
import numpy as np 
from PIL import Image
import math
import time
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
videoDevice = ALProxy('ALVideoDevice', "nao.local.", 9559)

motion = ALProxy("ALMotion","nao.local.",9559)
textProxy = ALProxy("ALTextToSpeech","nao.local.",9559)

AL_kTopCamera = 1
AL_kQVGA = 2           
AL_kBGRColorSpace = 13
captureDevice = videoDevice.subscribeCamera("scattqweqaaa", 0,1, 13, 30)

videoDevice.setParameter(0,11, 0);
videoDevice.setParameter(0,17, 2300);
while True:
    message = socket.recv()
    print(message)
    if(message == "acquisisci"):
            result = videoDevice.getImageRemote(captureDevice)
            im = Image.frombytes("RGB", (result[0], result[1]), result[6])
            cvim = np.array(im)
            frame_enc = cv2.imencode(".jpg", cvim)[1].tobytes()
            socket.send(frame_enc)
    if(message == "inizializzazione"):
            names = ["HeadPitch"]
            angle_rad = [math.radians(29)]
            motion.setStiffnesses("Body", 1.0)
            motion.setAngles(names, angle_rad, 0.1)
            socket.send("fatto")
    if(message == "alzaTesta"):
            names = ["HeadPitch"]
            angle_rad = [math.radians(0)]
            motion.setStiffnesses("Body", 1.0)
            motion.setAngles(names, angle_rad, 0.1)
            socket.send("fatto")
    if("riproduci_" in message):
        messaggio = message.split("_")[1]
        textProxy.say(messaggio)
        socket.send("fatto")