from tape_tracker.utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
import datetime
import argparse


class TapeTracker:
    def __init__(self, input_q, output_q, score_thresh=0.9, max_det=1, num_workers=4):
        self.threshold = score_thresh
        self.max_det = 1
        self.input_q = input_q
        self.output_q = output_q
        self.cap_params = self._init_cap_prop()

        self.pool = self._initialise_workers(num_workers, self.cap_params)

        # Other variables
        self.start_time = datetime.datetime.now()
        self.num_frames = 0
        self.fps = 0
        self.index = 0
    
    def _init_cap_prop(self):
        cap_params = {}
        cap_params['im_width'], cap_params['im_height'] = 576, 432
        cap_params['score_thresh'] = self.threshold
        cap_params['num_tapes_detect'] = self.max_det
        print(cap_params)
        return cap_params

    def _initialise_workers(self, num_workers, cap_params):
        # spin up workers to paralleize detection.
        pool = Pool(num_workers, self.worker,
                    (self.input_q, self.output_q, cap_params,))
        return pool


    def worker(self, input_q, output_q, cap_params):
        print(">> loading frozen model for worker")
        detection_graph, sess = detector_utils.load_inference_graph()
        sess = tf.Session(graph=detection_graph)  
        while True:
            #print("> ===== in worker loop, frame ", frame_processed)
            frame = input_q.get()

            if (frame is not None):
                # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                # while scores contains the confidence for each of these boxes.
                # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

                boxes, scores = detector_utils.detect_objects(
                    frame, detection_graph, sess)
                # draw bounding boxes
                detector_utils.draw_box_on_image(
                    self.cap_params['num_tapes_detect'], self.cap_params["score_thresh"],
                    scores, boxes, self.cap_params['im_width'], self.cap_params['im_height'],
                    frame)
                # add frame annotated with bounding box to queue
                output_q.put(frame)
            else:
                output_q.put(frame)
        sess.close()
    
    def close(self):
        self.pool.terminate()

