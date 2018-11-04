from tape_tracker.utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import threading


class TapeTrackerThread(threading.Thread):
    def __init__(self, input_q, output_q, score_thresh=0.9, max_det=1):
        self.threshold = score_thresh
        self.max_det = 1
        self.input_q = input_q
        self.output_q = output_q
        self.cap_params = self._init_cap_prop()

        # Other variables
        self.num_frames = 0
        self.fps = 0
        self.index = 0
        threading.Thread.__init__(self)
        print(">> loading frozen model for worker")
        self.detection_graph, self.sess = detector_utils.load_inference_graph()
        self.sess = tf.Session(graph=self.detection_graph)  

    def _init_cap_prop(self):
        cap_params = {}
        cap_params['im_width'], cap_params['im_height'] = 576, 432
        cap_params['score_thresh'] = self.threshold
        cap_params['num_tapes_detect'] = self.max_det
        print(cap_params)
        return cap_params

    def run (self):
        while True:
            frame = self.input_q.get()
            bb_coordinates = None
            if (frame is not None):
                # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                # while scores contains the confidence for each of these boxes.
                # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

                boxes, scores = detector_utils.detect_objects(
                    frame, self.detection_graph, self.sess)
                # draw bounding boxes
                bb_coordinates = detector_utils.draw_box_on_image(
                    self.cap_params['num_tapes_detect'], self.cap_params["score_thresh"],
                    scores, boxes, self.cap_params['im_width'], self.cap_params['im_height'],
                    frame)
                # add frame annotated with bounding box to queue
                self.output_q.put((frame, bb_coordinates))
            else:
                self.output_q.put((frame, bb_coordinates))
        
    def close (self):
        self.sess.close()


