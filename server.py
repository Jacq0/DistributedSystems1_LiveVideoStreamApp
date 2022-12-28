from flask import Flask, render_template, Response
from camera import Camera
from threading import Thread

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('page.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed(): 
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run()