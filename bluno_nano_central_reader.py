
# coding: utf-8

# In[1]:


from time import sleep
import serial # you'll need to install pySerial library 

ser = serial.Serial('/dev/cu.usbmodem1411', 115200) # change according to port, keep baud rate at 115200
print(ser.name) # verify that you've connected to port


# In[ ]:


# infinite while loop to read line by line
str_list = ["", ""]
print("Measuring... click encoder to get final measurement")

while True:
    ser_line = str(ser.readline()) # read line
    
    # store useful parts of previous two lines in list
    str_list[1] = str_list[0]
    str_list[0] = ser_line[2:-5]
    
    print(str_list)

    # check if final output received, if so, print useful part of previous line
    if (str_list[0] == "d" and str_list[1] != 's'): 
        print("Final Measurement: " + str_list[1])
        str_list = ["", ""] # reinitialise str_list
        print("Click encoder to start another measurement")
        


# In[ ]:


ser.close() # close port

