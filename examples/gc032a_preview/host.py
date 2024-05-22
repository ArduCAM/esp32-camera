# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
This code reads data from the serial port and displays it as an image preview.
It supports three formats of images (96x96 Grayscale, 240x320 Grayscale, 240x320 YUV).
The program also allows the user to select the image format.
"""

import serial
import serial.tools.list_ports
import numpy as np
import cv2
import getopt
import sys

np.set_printoptions(formatter={'int':hex})

# Convert YUV format to an image
def YUVToMat(data,Width,Height, color_mode):
    codeMap = {
        0:cv2.COLOR_YUV2BGR_YUYV,
        1:cv2.COLOR_YUV2BGR_YVYU,
        2:cv2.COLOR_YUV2BGR_UYVY,
        3:cv2.COLOR_YUV2BGR_Y422,
        4:cv2.COLOR_YUV2BGR_YUY2,
    }
    image = np.frombuffer(data, np.uint8).reshape(Height, Width, 2)
    return cv2.cvtColor(image,codeMap[color_mode])

# Convert RGB565 format to an image
def RGB565ToMat(data,Width,Height):
    arr = np.frombuffer(data,dtype='<u2').astype(np.uint32)
    arr = ((arr & 0xFF00) >> 8) + ((arr & 0x00FF) << 8)
    arr = 0xFF000000 + ((arr & 0xF800) << 8) + ((arr & 0x07E0) << 5) + ((arr & 0x001F) << 3)
    arr.dtype = np.uint8
    image = arr.reshape(Height,Width,4)
    return image

# Display the image preview
def preview_image(data,frame_len):

    print('preview\n')

    if(frame_len == 76804):
        image =  np.frombuffer(data,dtype =np.uint8).reshape(320,240)
        cv2.imshow("preview", image)
    elif(frame_len == 9220):
        image =  np.frombuffer(data,dtype =np.uint8).reshape(96,96)
        cv2.imshow("preview", image)
    elif(frame_len == 153604):
        image = RGB565ToMat(data,320,240)
        cv2.imshow("preview", image)
    cv2.waitKey(1) 

# Print the usage instructions
def usage(argv0):
    print("Usage: python "+argv0+" [options]")
    print("Available options are:")
    print(" -f        Choose the picture format,1: 240x320 YUV 2: 240x320 Grayscale 9:96x96 Grayscale")

# Select the serial port to use
def select_serial_port():
    ports_list = list(serial.tools.list_ports.comports())
    print("可用的串口设备如下：")
    idx = 0
    for comport in ports_list:
        print("[{}]:".format(idx),comport.description,comport.device)
        idx+=1
    index = input("please seletc: ")
    return ports_list[int(index)].device

if __name__=="__main__":
    # Check if the input argument is valid
    if(len(sys.argv) < 2):
        usage(sys.argv[0])
        sys.exit()

    # Get the input argument
    opts, arg = getopt.getopt(sys.argv[1:], 'hf:', ['help'])
    for option, value in opts:
        if option in ['-f']:
            if value.isdigit() == True:
                select_format = int(value)
            else:
                usage(sys.argv[0])
                sys.exit()
        else:
            usage(sys.argv[0])
            sys.exit()

    # Select the image format
    if(select_format == 1):
        send_cmd = b'\xfa\x09\xff'
    elif(select_format == 2):
        send_cmd = b'\xfa\x03\xff'
    elif(select_format == 3):
        send_cmd = b'\xfa\x01\xff'

    # Open the serial port
    ser = serial.Serial(select_serial_port(), 921600,timeout=0.1)
    #ser = serial.Serial("/dev/ttyACM0", 921600,timeout=0.1)

    if ser.isOpen():                        
        print("打开%s串口成功。"%ser.name)
    else:
        print("打开%s串口失败。"%ser.name)
    ser.write(send_cmd)
    data = b''
    data_len = 0
    frame_len = 0
    time_cnt = 0
    head_cnt = 0
    head = [b'\xff',b'\xff',b'\x00']

    # Read data from the serial port and display it as an image preview
    while True:
        rec = ser.read(size=1)
        if(rec == b''):
            time_cnt += 1
            if(time_cnt > 5):
                ser.write(send_cmd)
                data_len = 0
                data = b''
            continue
        time_cnt = 0
        data += rec
        data_len += 1
        if(head_cnt < 3):
            if(rec == head[head_cnt]):
                print(head[head_cnt])
                head_cnt+=1
            else:
                head_cnt = 0
        elif(head_cnt == 3) :
            if(rec == b'\03'):
                frame_len = 76804
            elif (rec == b'\t'):
                frame_len = 9220
            elif (rec == b'\01'):
                frame_len = 153604
            print(rec)
            head_cnt = 0
            
        if(data_len  == frame_len):
            preview_image(data[4:],frame_len)
            data_len = 0
            data = b''
            # ser.write(b'\xfa\x03\xff')

    # Close the serial port
    ser.close()