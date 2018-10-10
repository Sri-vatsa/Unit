import cv2
#from tf_pose_estimation.run_v3 import PoseEstimator

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        #self.estimator = PoseEstimator()

    #def _configure(self):
    #    self.estimator.configure()

    
    def __del__(self):
        self.video.release()
    
    def get_frame(self, estimator):
        success, image = self.video.read()
        img = cv2.resize(image, (720, 480)) 
        img = cv2.flip( img, 1)
        img = estimator.predict(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()