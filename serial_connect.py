import serial
import threading
import logging

class measurementSerialThread(threading.Thread):
  def __init__(self, dataQ, errQ, port=None, baudrate=None):
    self.logger = logging.getLogger('blunoSerialThread')
    self.logger.debug('initializing')
    threading.Thread.__init__(self)
    if baudrate is None:
      self.baudrate = 115200
    else:
      self.baudrate = baudrate
    self.ser = serial.Serial("COM3", self.baudrate) #/dev/cu.usbmodem1411
    print(self.ser.name)
    #if port is None:
    #  self.ser.port = "/dev/cu.usbmodem1411" #"/dev/tty.usbserial-A6004amR"
    #else:
    # self.ser.port = port
    #self.ser.flushInput()
    self.readCount = 0
    self.sleepDurSec = 5
    self.waitMaxSec = self.sleepDurSec * self.ser.baudrate / 10
    self.dataQ = dataQ
    self.errQ = errQ
    self.keepAlive = True
    self.stoprequest = threading.Event()
    self.setDaemon(True)
    self.dat = None
    self.inputStarted = False
    self.test = 1
  def run(self):
    self.logger.debug('running')
    
    dataIn = False
    while not self.stoprequest.isSet():
      #if not self.isOpen():
      #  self.connectForStream()
      
      str_list = ["", ""]
      while self.keepAlive:
        dat = self.readline()
        print(dat)
        # store useful parts of previous two lines in list
        str_list[1] = str_list[0]
        str_list[0] = dat[2:-5]
        #print(str_list)
        # check if final output received, if so, print useful part of previous line
        if (str_list[0] == "d" and str_list[1] != 's' and str_list[1] != ""): 
            print("Final Measurement: " + str_list[1])
            
            #Process data
            measurement = int(str_list[1])/12.0

            self.dataQ.put(str(measurement))
            str_list = ["", ""] # reinitialise str_list
        if not self.inputStarted:
          self.logger.debug('reading')
        self.inputStarted = True
      self.dat.close()
      self.close()
      self.join_fin()
    
  def join_fin(self):
    self.logger.debug('stopping')
    self.stoprequest.set()

  def connectForStream(self, debug=True):
    '''Attempt to connect to the serial port and fail after waitMaxSec seconds'''
    self.logger.debug('connecting')
    if not self.isOpen():
      self.logger.debug('Port not open, trying to open')
      try:
        self.open()
      except serial.serialutil.SerialException:
        self.logger.debug('Unable to use port ' + str(self.ser.port) + ', please verify and try again')
        return
    while self.readline() == '' and self.readCount < self.waitMaxSec and self.keepAlive:
        self.logger.debug('reading initial')
        self.readCount += self.sleepDurSec
        if not self.readCount % (self.ser.baudrate / 100):
          self.logger.debug("Verifying measurement data..")
          #some sanity check

    if self.readCount >= self.waitMaxSec:
        self.logger.debug('Unable to read measuremnet data...')
        self.close()
        return False
    else:
      self.logger.debug('Measurement data is streaming...')

    return True

  def isOpen(self):
    self.logger.debug('Open? ' + str(self.ser.isOpen()))
    return self.ser.isOpen()

  def open(self):
    self.ser.open()

  def stopDataAquisition(self):
    self.logger.debug('Falsifying keepAlive')
    self.keepAlive = False

  def close(self):
    self.logger.debug('closing')
    self.stopDataAquisition()
    self.ser.close()

  def write(self, msg):
    self.ser.write(msg)

  def readline(self):
    output_data = str(self.ser.readline())
    if output_data is "":
        return None
    else:
        return output_data

  def get_reading(self):
      if not self.dataQ.empty:
        return self.dataQ.get_nowait()
      else:
        return None
