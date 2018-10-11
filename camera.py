import cv2
import threading
#from tf_pose_estimation.run_v3 import PoseEstimator

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.stopped = False
        #self.estimator = PoseEstimator()

    #def _configure(self):
    #    self.estimator.configure()

    
    def __del__(self):
        self.video.release()
    
    def start(self):
        video_thread = threading.Thread(target=self.update)
        video_thread.daemon = True
        video_thread.start()
        return self
    
    def update(self):
        print("read")
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.video.read()

    def read(self, estimator):
        img = cv2.resize(self.frame, (432, 384)) 
        img = cv2.flip( img, 1)
        img = estimator.predict(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    
    '''
    def get_frame(self, estimator):
        success, image = self.video.read()
        img = cv2.resize(image, (432, 384)) 
        img = cv2.flip( img, 1)
        img = estimator.predict(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    '''