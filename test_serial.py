from serial_connect import measurementSerialThread
import queue

q_in = queue.Queue()
q_out = queue.Queue()

ser = measurementSerialThread(q_in, q_out)
