import argparse
import logging
import sys
import time
import glob
from math import sqrt, pow

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

body_mapping = ["nose", "neck", "R-shoulder", "R-elbow", "R-wrist", "L-shoulder", "L-elbow", "L-wrist", "R-hip", "R-knee", "R-ankle", "L-hip", "L-knee", "L-ankle", "L-eye", "R-eye", "L-ear", "R-ear"]

PIXEL_TO_DIST = 170.0/350 # represents the equivalent of 1 pixel in real distance (cm)

'''
Converts real height of a person into pixels
'''
def set_real_to_virtual(real_height, pixel_height):
    PIXEL_TO_DIST = real_height/pixel_height

'''
Compute the euclidean distance between two points
'''
def _compute_euclid_dist(x1, y1, x2, y2):
    
    distance = sqrt((pow(x2-x1, 2)+pow(y2-y1, 2)))

    return distance

'''
Compute the linear distance of arms of one person
Param: dictionary of key body joints
Return: real length of person arms
''' 
def compute_arm_length_for_one(body_measurements):

    left_arm_length, right_arm_length = None, None

    # compute the length of right arm
    if body_measurements.get("R-shoulder") and body_measurements.get("R-elbow") and body_measurements.get("R-wrist"):
    
        rShoulder = body_measurements["R-shoulder"]
        rElbow = body_measurements["R-elbow"] 
        rWrist = body_measurements["R-wrist"] 

        right_arm_length = PIXEL_TO_DIST*(_compute_euclid_dist(rShoulder[0],rShoulder[1],rElbow[0],rElbow[1]) + _compute_euclid_dist(rElbow[0],rElbow[1],rWrist[0],rWrist[1]))

    # compute the length of left arm
    if body_measurements.get("L-shoulder") and body_measurements.get("L-elbow") and body_measurements.get("L-wrist"):
        lShoulder = body_measurements["L-shoulder"] 
        lElbow = body_measurements["L-elbow"] 
        lWrist = body_measurements["L-wrist"] 

        left_arm_length = PIXEL_TO_DIST*(_compute_euclid_dist(lShoulder[0],lShoulder[1],lElbow[0],lElbow[1]) + _compute_euclid_dist(lElbow[0],lElbow[1],lWrist[0],lWrist[1]))

    return (left_arm_length, right_arm_length)


'''
Compute the linear distance of legs of one person
Param: dictionary of key body joints
Return: real length of person legs
''' 
def compute_leg_length_for_one(body_measurements):

    left_leg_length, right_leg_length = None, None

    # compute the length of right arm
    if body_measurements.get("R-hip") and body_measurements.get("R-knee") and body_measurements.get("R-ankle"):
    
        rHip = body_measurements["R-hip"]
        rKnee = body_measurements["R-knee"] 
        rAnkle = body_measurements["R-ankle"] 

        right_leg_length = PIXEL_TO_DIST*(_compute_euclid_dist(rHip[0],rHip[1],rKnee[0],rKnee[1]) + _compute_euclid_dist(rKnee[0],rKnee[1],rAnkle[0],rAnkle[1]))

    # compute the length of left arm
    if body_measurements.get("L-hip") and body_measurements.get("L-knee") and body_measurements.get("L-ankle"):
        lHip= body_measurements["L-hip"] 
        lKnee = body_measurements["L-knee"] 
        lAnkle = body_measurements["L-ankle"] 

        left_leg_length = PIXEL_TO_DIST*(_compute_euclid_dist(lHip[0],lHip[1],lKnee[0],lKnee[1]) + _compute_euclid_dist(lKnee[0],lKnee[1],lAnkle[0],lAnkle[1]))

    return (left_leg_length, right_leg_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image-path', type=str, default='./images/test_2/fat_man.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    # import images from video broken into frames
    #image_collection = [cv2.imread(file) for file in glob.glob(args.image_path)] 
    #print(len(image_collection))

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    sum_arm_lengths = [0, 0]
    sum_leg_lengths = [0, 0]

    count_arm = 0
    count_leg = 0

    image = cv2.imread(args.image_path)

    # estimate human poses from a single image !
    #image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image cannot be read, path=%s' % args.image)
        sys.exit(-1)
    #t = time.time()

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    #elapsed = (time.time() - t)/runs 
    #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image, body_coordinates = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
    readable_body_coordinates = []
    arm_lengths_all = []
    leg_lengths_all = []
    for human in range(len(body_coordinates)):
        each_human_coordinates = {}
        for part_id in range(18):
            if body_coordinates[human].get(part_id):
                each_human_coordinates[body_mapping[part_id]] = body_coordinates[human][part_id]   
            
        # Collate all body points by person in an array of dictionaries
        readable_body_coordinates.append(each_human_coordinates)
            
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
            
    ## Print measurements to screen
    print("Arm measurements for each person (Left, Right):")
    print(arm_lengths_all)
    print("Leg measurements for each person (Left, Right):")
    print(leg_lengths_all)

    #TODO modify adding sum to multiple people
        
    if arm_lengths_all[0] is not None: 
        sum_arm_lengths[0] += arm_lengths_all[0][0]
        sum_arm_lengths[1] += arm_lengths_all[0][1]
        count_arm += 1

    if leg_lengths_all[0] is not None:
        sum_leg_lengths[0] += leg_lengths_all[0][0]
        sum_leg_lengths[1] += leg_lengths_all[0][1]
        count_leg += 1
        
    #sum_arm_lengths[:] = [x/count_arm for x in sum_arm_lengths]
    #sum_leg_lengths[:] = [x/count_leg for x in sum_leg_lengths]

    #print("#########")
    #print("Final lengths:")
    #print("#########")
    #print("Arm measurements for each person (Left, Right):")
    #print(sum_arm_lengths)
    #print("Leg measurements for each person (Left, Right):")
    #print(sum_leg_lengths)
    
    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imwrite("images/test.jpeg",image)

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    fig.add_subplot()
    # show network output
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    #print(np.array2string(tmp, threshold=np.nan, max_line_width=np.nan)) # Print full np array REMOVE IN DEPLOYMENT
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    '''
    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    '''
    plt.show()
