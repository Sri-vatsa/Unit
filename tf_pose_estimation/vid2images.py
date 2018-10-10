import cv2

vidcap = cv2.VideoCapture('./images/originals/normal.MOV')
success,image = vidcap.read()
print(vidcap.get(cv2.CAP_PROP_FPS))
count = 0
while success:
  cv2.imwrite("./images/images_from_video_normal/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  if image is not None:
    image = cv2.resize(image, (368, 656)) 
  print('Read a new frame: ', success)
  count += 1

vidcap.release()