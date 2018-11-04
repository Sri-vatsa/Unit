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
from tape_tracker.tape_tracker import TapeTrackerThread

app = Flask(__name__, static_url_path='')

estimator = PoseEstimator()
estimator.configure()
dataQ = queue.Queue()
errQ = queue.Queue()
objd_in_q = queue.Queue()
objd_out_q = queue.Queue()
measurements_consol = {}
ser = measurementSerialThread(dataQ, errQ, baudrate=115200)
ser.daemon = True
ser.start()
print("Measurement thread started")

tracker = TapeTrackerThread(objd_in_q, objd_out_q)
tracker.daemon = True
tracker.start()
print("Tracker is initialised")

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

@app.route('/get_readings' , methods=['GET','POST'])
def get_readings():
    #data = request.form['keyword']
    data = get_bluno_data()
    measurements_consol["data"] = data
    resp = make_response(json.dumps(measurements_consol))
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/next', methods=['POST'])
def next():
  estimator.state_next()
  print(measurements_consol[0])
  resp = make_response(estimator.state)
  resp.status_code = 200
  resp.headers['Access-Control-Allow-Origin'] = '*'
  return resp

@app.route('/reset', methods=['POST'])
def reset():
  estimator.state_reset()
  resp = make_response("Success")
  resp.status_code = 200
  resp.headers['Access-Control-Allow-Origin'] = '*'
  return resp

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
    count = 0
    while True:
        if camera.stopped:
            break
        #frame = camera.get_frame(estimator)
        #get_bluno_data()

        curr_state = estimator.state

        img = camera.read()

        # estimate
        img_est, measurements = estimator.predict(img)
        measurements_consol[0] = measurements

        # obj det
        objd_in_q.put(img)
        (img_det, bb_coord) = objd_out_q.get()
        
        if len(bb_coord) != 0 and count > 10:
          estimator.configure_tracker(bb_coord[0])
          count = 0
          print(bb_coord[0])

        count = count + 1

        if curr_state is not "reference":
          byte_frame = img_to_bytes(img_est)
        else:
          byte_frame = img_to_bytes(img_det)

        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n\r\n')

def img_to_bytes(img):
  ret, jpeg = cv2.imencode('.jpg', img)
  return jpeg.tobytes()

def get_bluno_data():
  measurement = read_measurement_from_serial()
  return measurement

def read_measurement_from_serial():
  reading = "No reading yet"
  if not dataQ.empty():
    reading = dataQ.get_nowait()
  return reading


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True, ssl_context='adhoc')