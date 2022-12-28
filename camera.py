from imutils import paths
import cv2
import face_recognition as fr
import pickle
import os
import numpy as np

class Camera():
    def __init__(self):        
        self.cap = cv2.VideoCapture(0)  # Prepare the camera...
        self.classifier = cv2.CascadeClassifier('classifier\haarcascade_frontalface_default.xml') #haarscascade classifier
        print("Camera warming up ...")
        self.encode_faces()

    def get_frame(self):
       ret, frame = self.cap.read()
       frame = self.detect_faces(frame) #do facial detection on frame

       ret2, buffer = cv2.imencode('.jpg', frame)
    
       frame = buffer.tobytes()
       if ret and ret2:  # frame captures without any errors...
           pass
        
       return frame

    def detect_faces(self, frame):
        data_enc = pickle.loads(open(os.getcwd() + '\\encodings.pickle', 'rb').read())

        print(data_enc)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_enc = fr.face_encodings(frame_rgb)

        for enc in data_enc:
            for face in frame_enc:
                results = fr.compare_faces(np.array(enc['encoding']), face)

                if results[0] == True:
                    print("Found a recognised face " +  enc['name'])

        return frame

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
