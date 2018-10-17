#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response, send_from_directory, request, make_response
from camera import VideoCamera
from tf_pose_estimation.run_v3 import PoseEstimator
import threading, queue
import cv2
import json
from serial_connect import measurementSerialThread

app = Flask(__name__, static_url_path='')

estimator = PoseEstimator()
estimator.configure()
dataQ = queue.Queue()
errQ = queue.Queue()
#ser = measurementSerialThread(dataQ, errQ, baudrate=115200)
#ser.daemon = True
#ser.start()
print("Measurement thread started")

#@app.before_first_request
#def configure():
#    estimator.configure()

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/images/<path:path>')
def send_img(path):
    return send_from_directory('images', path)

@app.route('/')
def index():
    return render_template('index.html')

'''
@app.route('/get_readings' , methods=['GET','POST'])
def get_readings():
    #data = request.form['keyword']
    data = get_bluno_data()
    #print(data)
    resp = make_response(json.dumps(data))
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
'''

@app.route('/configure_tracker', methods=['POST'])
def configure_tracker():
    data = request.form
    bounding_box = json.loads(data["bb"])
    estimator.configure_tracker(bounding_box)
    resp = make_response("Success")
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera().start()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        if camera.stopped:
            break
        #frame = camera.get_frame(estimator)
        #get_bluno_data()
        frame = camera.read(estimator)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def get_bluno_data():
  measurement = read_measurement_from_serial()
  return measurement

def read_measurement_from_serial():
  reading = "No reading yet"
  if not dataQ.empty():
    reading = dataQ.get_nowait()
  return reading


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)