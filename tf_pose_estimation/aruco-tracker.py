import numpy as np
import cv2
import cv2.aruco as aruco
import glob

cap = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calib_images/*.jpg')
 
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


while (True):
    ret, frame = cap.read()
    
    #Rectangle Coordinates
    RectTopLeftX = 200
    RectTopLeftY = 150
    RectBottomRightX = 300
    RectBottomRightY = 250
    #draw rectangle
    cv2.rectangle(frame, (RectTopLeftX,RectTopLeftY), (RectBottomRightX, RectBottomRightY), (200,0,0), 5)
    
    
    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

    if np.all(ids != None):
        rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[0], 0.05, mtx, dist) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        aruco.drawAxis(frame, mtx, dist, rvec[0], tvec[0], 0.1) #Draw Axis
        aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers
        
        #Obtain x,y screen coordinates of the centre of aruco marker
        x = (corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]) / 4
        y = (corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]) / 4
        print(str(x)+ "," + str(y))

        if x > RectTopLeftX and x < RectBottomRightX and y > RectTopLeftY and y < RectBottomRightY:
            Red = 0
            Green = 200
            Blue = 0
        else:
            Red = 200
            Green = 0
            Blue = 0
                        
        #draw rectangle
        cv2.rectangle(frame, (RectTopLeftX,RectTopLeftY), (RectBottomRightX, RectBottomRightY), (Red,Green,Blue), 5)

        #text for guidance system
        if x < RectTopLeftX:
            text1 = "Left"
        elif x > RectBottomRightX:
            text1= "Right"
        else:
            text1 = "Correct"

        if y < RectTopLeftY:
            text2 = "Lower"
        elif y > RectBottomRightY:
            text2 = "Higher"
        else:
            text2 = "Correct"
            
        cv2.putText(frame, "X Adj: " + str(text1) + ", Y Adj: " + str(text2) , (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        if text1 == "Correct" and text2 == "Correct":
            cv2.putText(frame, "Hold still, measuring..." , (0,30), font, 1, (0,255,0),2,cv2.LINE_AA)

        ###### DRAW ID #####
        #cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
