import cv2
class Camera():
    def __init__(self):        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Prepare the camera...
        print("Camera warming up ...")


    def get_frame(self):
       ret, img = self.cap.read()
       ret2, buffer = cv2.imencode('.jpg', img)

       img = buffer.tobytes()
       #if ret2:  # frame captures without errors...
           #pass
        
       return img

    def release_camera(self):
        self.cap.release()
