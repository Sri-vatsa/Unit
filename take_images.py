import cv2

cam = cv2.VideoCapture(0)

counter = 0
while True:
    ret, img = cam.read()
    img = cv2.resize(img, (576, 432))
    cv2.imshow('frame', img)
    cv2.imwrite("./data/img%s.jpg"%counter, img)
    cv2.waitKey(1)
    counter = counter + 1
    if counter == 10:
        break