from tape_tracker import TapeTracker
from multiprocessing import Queue
import cv2
import time

input_q = Queue(20)
output_q = Queue(20)
cam = cv2.VideoCapture(0)
tracker = TapeTracker(input_q, output_q)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (576, 432))
    input_q.put(frame)

    output_frame = output_q.get()
    if output_frame is not None:
        cv2.imshow('frame', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

tracker.close()
cv2.destroyAllWindows()