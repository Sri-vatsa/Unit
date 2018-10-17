import argparse
import logging
import sys
import time
import glob
from math import sqrt, pow, pi, sin, cos

sys.path.append("./tf_pose_estimation/")

from tf_pose import common
import cv2
import cv2.aruco as aruco
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.disabled = True
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class PoseEstimator():
    def __init__(self, model_type='mobilenet_thin', width=144, height=128):
        #self.cam_id = cam
        #self.cam = None
        self.model_type = model_type
        self.width = width
        self.height = height
        self.resize_out_ratio = 4.0
        self.model = None
        self.mtx = None
        self.dist = None
        
        self.initBB = None #(176, 89, 72, 115)
        self.tracker = None
        self.trackerHasStarted = False

        self.body_mapping = ["nose", "neck", "R-shoulder", "R-elbow", "R-wrist", "L-shoulder", "L-elbow", "L-wrist",
                "R-hip", "R-knee", "R-ankle", "L-hip", "L-knee", "L-ankle", "L-eye", "R-eye", "L-ear", "R-ear"]

        # represents the equivalent of 1 pixel in real distance (cm)
        self.PIXEL_TO_DIST = 170.0/350
        self.fps_time = 0
        self.CALIBRATION_IMAGES_PATH = 'tf_pose_estimation/calib_images/*.jpg'

    '''
    Set Aruco parameters
    '''
    def configure_aruco(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        return objp, criteria

    '''
    Camera calibration for aruco
    '''
    def calibrate_camera(self, objp, criteria):
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(self.CALIBRATION_IMAGES_PATH)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist

    def aruco_tracking(self, image, mtx, dist, rectangle_corners):
        
        #print(rectangle_corners)
        # Rectangle Coordinates
        if rectangle_corners is not None:
            RectTopLeftX = rectangle_corners[0][0]
            RectTopLeftY = rectangle_corners[0][1]
            RectBottomRightX = rectangle_corners[1][0]
            RectBottomRightY = rectangle_corners[1][1]
            # draw rectangle
            # cv2.rectangle(image, (RectTopLeftX,RectTopLeftY), (RectBottomRightX, RectBottomRightY), (200,0,0), 5)

            # operations on the frame come here
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()

            # lists of ids and the corners beloning to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            '''
            print("#####################")
            print("Corners")
            print(corners)
            print("#####################")
            print("Rejected points")
            print(rejectedImgPoints)
            print("#####################")
            '''
            font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

            if np.all(ids != None):
                # Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[0], 0.05, mtx, dist)
                # (rvec-tvec).any() # get rid of that nasty numpy value array error

                aruco.drawAxis(image, mtx, dist,
                               rvec[0], tvec[0], 0.1)  # Draw Axis
                # Draw A square around the markers
                aruco.drawDetectedMarkers(image, corners)

                # Obtain x,y screen coordinates of the centre of aruco marker
                x = (corners[0][0][0][0] + corners[0][0][1][0] +
                     corners[0][0][2][0] + corners[0][0][3][0]) / 4
                y = (corners[0][0][0][1] + corners[0][0][1][1] +
                     corners[0][0][2][1] + corners[0][0][3][1]) / 4
                #print(str(x) + "," + str(y))

                if x > RectTopLeftX and x < RectBottomRightX and y > RectTopLeftY and y < RectBottomRightY:
                    Red = 0
                    Green = 200
                    Blue = 0
                else:
                    Red = 200
                    Green = 0
                    Blue = 0

                # draw rectangle
                cv2.rectangle(image, (RectTopLeftX, RectTopLeftY),
                              (RectBottomRightX, RectBottomRightY), (Red, Green, Blue), 5)

                # text for guidance system
                if x < RectTopLeftX:
                    text1 = "Left"
                elif x > RectBottomRightX:
                    text1 = "Right"
                else:
                    text1 = "Correct"

                if y < RectTopLeftY:
                    text2 = "Lower"
                elif y > RectBottomRightY:
                    text2 = "Higher"
                else:
                    text2 = "Correct"

                cv2.putText(image, "X Adj: " + str(text1) + ", Y Adj: " +
                            str(text2), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if text1 == "Correct" and text2 == "Correct":
                    cv2.putText(image, "Hold still, measuring...",
                                (0, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image

    '''
    Converts real height of a person into pixels
    '''

    def set_real_to_virtual(self, real_height, pixel_height):
        PIXEL_TO_DIST = real_height/pixel_height

    '''
    Compute the euclidean distance between two points
    '''

    def _compute_euclid_dist(self, x1, y1, x2, y2):

        distance = sqrt((pow(x2-x1, 2)+pow(y2-y1, 2)))

        return distance

    '''
    Compute the linear distance of arms of one person
    Param: dictionary of key body joints
    Return: real length of person arms
    '''

    def compute_arm_length_for_one(self, body_measurements):

        left_arm_length, right_arm_length = None, None

        # compute the length of right arm
        if body_measurements.get("R-shoulder") and body_measurements.get("R-elbow") and body_measurements.get("R-wrist"):

            rShoulder = body_measurements["R-shoulder"]
            rElbow = body_measurements["R-elbow"]
            rWrist = body_measurements["R-wrist"]

            right_arm_length = self.PIXEL_TO_DIST*(self._compute_euclid_dist(
                rShoulder[0], rShoulder[1], rElbow[0], rElbow[1]) + self._compute_euclid_dist(rElbow[0], rElbow[1], rWrist[0], rWrist[1]))

        # compute the length of left arm
        if body_measurements.get("L-shoulder") and body_measurements.get("L-elbow") and body_measurements.get("L-wrist"):
            lShoulder = body_measurements["L-shoulder"]
            lElbow = body_measurements["L-elbow"]
            lWrist = body_measurements["L-wrist"]

            left_arm_length = self.PIXEL_TO_DIST*(self._compute_euclid_dist(
                lShoulder[0], lShoulder[1], lElbow[0], lElbow[1]) + self._compute_euclid_dist(lElbow[0], lElbow[1], lWrist[0], lWrist[1]))

        return (left_arm_length, right_arm_length)

    def compute_arm_centre_left(self, body_measurements):

        result = None

        if body_measurements.get("L-shoulder") and body_measurements.get("L-elbow"):

            lShoulder = body_measurements["L-shoulder"]
            lElbow = body_measurements["L-elbow"]

            x = self.find_mid(lShoulder[0], lElbow[0])
            y = self.find_mid(lShoulder[1], lElbow[1])

            result = (x, y)

        return result

    def compute_arm_centre_right(self, body_measurements):

        result = None

        if body_measurements.get("R-shoulder") and body_measurements.get("R-elbow"):

            rShoulder = body_measurements["R-shoulder"]
            rElbow = body_measurements["R-elbow"]

            x = self.find_mid(rShoulder[0], rElbow[0])
            y = self.find_mid(rShoulder[1], rElbow[1])

            result = (x, y)

        return result

    def find_mid(self, x1, x2):
        return (x1 + x2)/2

    # TODO: Complete function
    # Compute angle in between arm and vertical
    # Return angle in degrees
    def compute_angle_of_left_arm(self, body_measurements):
        angle_deg = None
        if body_measurements.get("L-shoulder") and body_measurements.get("L-elbow"):

            lShoulder = body_measurements["L-shoulder"]
            lElbow = body_measurements["L-elbow"]
            third_point = [lShoulder[0], lElbow[1]]

            _angle_rad = self._compute_angle(lShoulder, lElbow, third_point)
            angle_deg = _angle_rad * 180.0 / pi
        return angle_deg

    # TODO: Complete function
    # Compute angle in between arm and vertical
    # Return angle in degrees
    def compute_angle_of_right_arm(self, body_measurements):
        angle_deg = None
        if body_measurements.get("R-shoulder") and body_measurements.get("R-elbow"):
            rShoulder = body_measurements["R-shoulder"]
            rElbow = body_measurements["R-elbow"]
            third_point = [rShoulder[0], rElbow[1]]

            _angle_rad = self._compute_angle(rShoulder, rElbow, third_point)
            angle_deg = _angle_rad * 180.0 / pi
        return angle_deg

    def _compute_angle(self, p0, p1, p2):

        p0 = np.asarray(p0)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        vec_1 = p1 - p0
        vec_2 = p2 - p0

        angle = np.arccos(np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))
        return angle

    def draw_angled_rec(self, x0, y0, width, height, angle, img):

        _angle = angle * pi / 180.0
        b = cos(_angle) * 0.5
        a = sin(_angle) * 0.5
        try:
            pt0 = (int(x0 - a * height - b * width),
                int(y0 + b * height - a * width))
            pt1 = (int(x0 + a * height - b * width),
                int(y0 - b * height - a * width))
            pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
            pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

            rectangle_corners = (pt1, pt3)

            cv2.line(img, pt0, pt1, (255, 255, 255), 2)
            cv2.line(img, pt1, pt2, (255, 255, 255), 2)
            cv2.line(img, pt2, pt3, (255, 255, 255), 2)
            cv2.line(img, pt3, pt0, (255, 255, 255), 2)
        except:
            return img, None

        return img, rectangle_corners

    '''
    Compute the linear distance of legs of one person
    Param: dictionary of key body joints
    Return: real length of person legs
    '''

    def compute_leg_length_for_one(self, body_measurements):

        left_leg_length, right_leg_length = None, None

        # compute the length of right arm
        if body_measurements.get("R-hip") and body_measurements.get("R-knee") and body_measurements.get("R-ankle"):

            rHip = body_measurements["R-hip"]
            rKnee = body_measurements["R-knee"]
            rAnkle = body_measurements["R-ankle"]

            right_leg_length = self.PIXEL_TO_DIST*(self._compute_euclid_dist(
                rHip[0], rHip[1], rKnee[0], rKnee[1]) + self._compute_euclid_dist(rKnee[0], rKnee[1], rAnkle[0], rAnkle[1]))

        # compute the length of left arm
        if body_measurements.get("L-hip") and body_measurements.get("L-knee") and body_measurements.get("L-ankle"):
            lHip = body_measurements["L-hip"]
            lKnee = body_measurements["L-knee"]
            lAnkle = body_measurements["L-ankle"]

            left_leg_length = self.PIXEL_TO_DIST*(self._compute_euclid_dist(
                lHip[0], lHip[1], lKnee[0], lKnee[1]) + self._compute_euclid_dist(lKnee[0], lKnee[1], lAnkle[0], lAnkle[1]))

        return (left_leg_length, right_leg_length)

    def is_tracker_in_pose_rect(self, tracker_coord, pose_rect_corners):
        if tracker_coord is None or pose_rect_corners is None:
            return False
        return 

    def csrt_tracking(self, frame, initBB):    
        
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = self.tracker.update(frame)
            #print(box)

            tracker_coord = None

            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)
                tracker_coord = (x,y,w,h)

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", "MIL"),
                ("Success", "Yes" if success else "No"),
            ]
            
            '''
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, 432 - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            '''
        return frame, tracker_coord

    def configure_tracker(self, bb):
        # kcf tracker setup
        #self.tracker = cv2.TrackerMIL_create()
        if self.trackerHasStarted:
            self.tracker = None
            print()
            self.tracker = cv2.TrackerCSRT_create()
            self.initBB = (bb[0], bb[1], bb[2], bb[3])
            self.trackerHasStarted = False
        else:
            self.initBB = (bb[0], bb[1], bb[2], bb[3])
        #print(self.initBB)

    # Main functions
    def configure(self):
        #self.cam = cv2.VideoCapture(self.cam_id)
        # camera calibration
        #objp, criteria = self.configure_aruco()
        #self.mtx, self.dist = self.calibrate_camera(objp, criteria)

        # CSRT tracker setup
        #self.tracker = cv2.TrackerCSRT_create()

        # kcf tracker setup
        self.tracker = cv2.TrackerCSRT_create()

        if self.width == 0 or self.height == 0:
            self.model = TfPoseEstimator(get_graph_path(self.model_type), target_size=(432, 368))
        else:
            self.model = TfPoseEstimator(get_graph_path(self.model_type), target_size=(self.width, self.height))

        #print("YAY Config was done")
        #print(self.mtx, self.dist, self.width, self.height)


    def predict(self, orig_image):
        #ret_val, orig_image = self.cam.read()
        # estimate human poses from a single image !
        # image = common.read_imgfile(args.image, None, None)
        if orig_image is None:
            logger.error('Webcam not working')
            sys.exit(-1)
        # t = time.time()
        image_with_lines = orig_image.copy()
        image = orig_image.copy()
        humans = self.model.inference(image_with_lines, resize_to_default=(self.width > 0 and self.height > 0), upsample_size=self.resize_out_ratio)
        # elapsed = (time.time() - t)/runs
        # logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

        image_with_lines, body_coordinates = TfPoseEstimator.draw_humans(image_with_lines, humans, imgcopy=False)
        pose_rect_corners = None
        tracker_coord = None
        arm_lengths_all = []
        leg_lengths_all = []
        # Process body data detected by model
        for human in range(len(body_coordinates)):
            each_human_coordinates = {}
            for part_id in range(18):
                if body_coordinates[human].get(part_id):
                    each_human_coordinates[self.body_mapping[part_id]] = body_coordinates[human][part_id]

            # Compute arm measurements for each person
            person_arm_lengths = self.compute_arm_length_for_one(each_human_coordinates)
            if person_arm_lengths[0] is not None and person_arm_lengths[1] is not None:
                arm_lengths_all.append(person_arm_lengths)
            else:
                arm_lengths_all.append(None)

            # Find centre of left arm and right arm
            left_centre = self.compute_arm_centre_left(each_human_coordinates)
            right_centre = self.compute_arm_centre_right(each_human_coordinates)

            #print(left_centre)

            left_arm_angle = self.compute_angle_of_left_arm(each_human_coordinates)
            right_arm_angle = self.compute_angle_of_right_arm(each_human_coordinates)

            # draw angled rectangle on left arm
            # TODO fix angle
            if left_centre is not None:
                image, pose_rect_corners = self.draw_angled_rec(
                    left_centre[0], left_centre[1], 50, 50, left_arm_angle, image)

            # draw angled rectangle on right arm
            # TODO fix angle
            if right_centre is not None:
                image, pose_rect_corners = self.draw_angled_rec(
                    right_centre[0], right_centre[1], 50, 50, right_arm_angle, image)

            #print("Image shape: " + str(image.shape))

        # TODO toggle between rectangle corners for different squares for different measurements
        #image = self.aruco_tracking(image, self.mtx, self.dist, pose_rect_corners)
        if not self.trackerHasStarted and self.initBB is not None:
            self.tracker.init(image, self.initBB)
            self.trackerHasStarted = True
        
        if self.trackerHasStarted and self.initBB is not None:
            image, tracker_coord = self.csrt_tracking(image, self.initBB)

        isTrackerInRightPlace = self.is_tracker_in_pose_rect(tracker_coord, pose_rect_corners)
        # Print measurements to screen
        #print("Arm measurements for each person (Left, Right):")
        #print(arm_lengths_all)
        # print("Leg measurements for each person (Left, Right):")
        # print(leg_lengths_all)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', image)
        self.fps_time = time.time()
        #cv2.waitKey(1)
        #if cv2.waitKey(1) == 27:
        #    break
        logger.debug('finished+')
        return image

    '''
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        parser.add_argument('--camera', type=int, default=0)
        parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

        parser.add_argument('--resize', type=str, default='144x128',
                            help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        args = parser.parse_args()

        # import images from video broken into frames
        cam = cv2.VideoCapture(args.camera)

        # camera calibration
        objp, criteria = configure_aruco()
        mtx, dist = calibrate_camera(objp, criteria)

        ret_val, image = cam.read()

        w, h = model_wh(args.resize)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

        # sum_arm_lengths = [0, 0]
        # sum_leg_lengths = [0, 0]

        # count_arm = 0
        # count_leg = 0

        
        while True:

            ret_val, orig_image = cam.read()
            # estimate human poses from a single image !
            # image = common.read_imgfile(args.image, None, None)
            if orig_image is None:
                logger.error('Webcam not working')
                sys.exit(-1)
            # t = time.time()
            image_with_lines = orig_image.copy()
            image = orig_image.copy()
            humans = e.inference(image_with_lines, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            # elapsed = (time.time() - t)/runs 
            # logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

            image_with_lines, body_coordinates = TfPoseEstimator.draw_humans(image_with_lines, humans, imgcopy=False)
            pose_rect_corners = None
            arm_lengths_all = []
            leg_lengths_all = []
            for human in range(len(body_coordinates)):
                each_human_coordinates = {}
                for part_id in range(18):
                    if body_coordinates[human].get(part_id):
                        each_human_coordinates[body_mapping[part_id]] = body_coordinates[human][part_id]   
                
                # Compute arm measurements for each person
                person_arm_lengths = compute_arm_length_for_one(each_human_coordinates)
                if person_arm_lengths[0] is not None and person_arm_lengths[1] is not None:
                    arm_lengths_all.append(person_arm_lengths)
                else:
                    arm_lengths_all.append(None)

                # Compute leg measurements for each person
                
        person_leg_lengths = compute_leg_length_for_one(each_human_coordinates)
        if person_leg_lengths[0] is not None and person_leg_lengths[1] is not None:
            leg_lengths_all.append(person_leg_lengths)
        else:
            leg_lengths_all.append(None)
    

                # Find centre of left arm and right arm
                left_centre = compute_arm_centre_left(each_human_coordinates)
                right_centre = compute_arm_centre_right(each_human_coordinates)

                print(left_centre)

                left_arm_angle = compute_angle_of_left_arm(each_human_coordinates)
                right_arm_angle = compute_angle_of_right_arm(each_human_coordinates)

                # draw angled rectangle on left arm
                # TODO fix angle
                if left_centre is not None:
                    image, pose_rect_corners = draw_angled_rec(left_centre[0], left_centre[1], 100, 100, left_arm_angle, image)

                # draw angled rectangle on right arm
                # TODO fix angle
                if right_centre is not None:
                    image, pose_rect_corners = draw_angled_rec(right_centre[0], right_centre[1], 100, 100, right_arm_angle, image)

                print("Image shape: " + str(image.shape))
            
            # TODO toggle between rectangle corners for different squares for different measurements
            image = aruco_tracking(image, mtx, dist, pose_rect_corners)

            # Print measurements to screen
            print("Arm measurements for each person (Left, Right):")
            print(arm_lengths_all)
            # print("Leg measurements for each person (Left, Right):")
            # print(leg_lengths_all)

            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')
            '''