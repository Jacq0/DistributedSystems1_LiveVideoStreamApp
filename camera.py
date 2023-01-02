from imutils import paths
import cv2
import face_recognition as fr
import pickle
import os
import numpy as np

class Camera():

    known_encodings = []
    frame_count=0

    def __init__(self):        
        self.cap = cv2.VideoCapture(0)  # Prepare the camera...
        self.set_cam_res(320, 240, self.cap)
        self.classifier = cv2.CascadeClassifier('classifier\haarcascade_frontalface_default.xml') #haarscascade classifier
        print("Camera warming up ...")
        self.encode_faces()
        self.known_encodings = self.load_faces()
        

    def get_frame(self):
       ret, frame = self.cap.read()
       frame = self.detect_faces(frame) #do facial detection on frame

       ret2, buffer = cv2.imencode('.jpg', frame)
    
       frame = buffer.tobytes()
       if ret and ret2:  # frame captures without any errors...
           pass
        
       return frame

    def detect_faces(self, frame):
        if self.frame_count%2 == 0: #only process every second frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = fr.face_locations(frame_rgb, model='hog')
            frame_enc = fr.face_encodings(frame_rgb, boxes)

            for i in range(0, len(frame_enc)):
                results = fr.compare_faces(self.known_encodings, frame_enc[i], 0.85)

                if results[0] == True:
                    print("Found a recognised face")
                    self.draw_rects(frame, boxes[i], True)
                else:
                    self.draw_rects(frame, boxes[i], False)
                    

        self.frame_count += 1
        return frame

    def load_faces(self):
        data_enc = pickle.loads(open(os.getcwd() + '\\encodings.pickle', 'rb').read())
        encodings = []

        for enc in data_enc:
            encodings.append(enc['encoding'])

        print(data_enc, " KNOWN ENCODINGS")
        
        return encodings

    def encode_faces(self):
        encoding_data = []

        image_paths = list(paths.list_images(os.getcwd() + '\\face_images'))

        for (i, image_path) in enumerate(image_paths):
            image, name = cv2.imread(image_path), image_path.split('\\')[-1]
            imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #dlib expects RGB while opencv uses BGR 
            boxes = fr.face_locations(imgrgb, model='hog')

            for enc in fr.face_encodings(imgrgb, boxes):

                data = {'encoding': enc, 'name': name}
                encoding_data.append(data)

        f = open(os.getcwd() + '\\encodings.pickle', 'wb')
        f.write(pickle.dumps(encoding_data))
        f.close()
        
    def release_camera(self):
        self.cap.release()
    
    def set_cam_res(self, w, h, cap):
        cap.set(3, w)
        cap.set(4, h)

    def draw_rects(self, frame, box, recognised):
        color = (0,0,255)

        if recognised:
            color = (0,255,0)

        cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), color, 2)
