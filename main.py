# -*- coding: utf-8 -*-
"""
Created on Sat May 16 06:04:18 2020

@author: Abdul Basit
"""

"""People Counter Application Using INTEL OPENVINO"""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# IMPORTING REQUIRED LIBRARIES FOR THE PROJECT
import os
import sys
import time
import socket # In order to connect to the MQTT server
import json
import cv2 # Importing OPENCV


#Flexible Event Logging System
import logging as log
#Python library for working with MQTT
import paho.mqtt.client as mqtt
# In order to hold all the information necessary to parse the command line into Python data types
from argparse import ArgumentParser
# Import Network from inference
from inference import Network


# MQTT server environment variables
# Imported socket library to connect to the MQTT server
# Get the IP address and set the port for communicating with the MQTT server
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
#The above code lines set the IP address and port, as well as the keep alive interval
#The keep alive interval is used so that the server and client will communicate every 60 seconds to confirm their connection is still open

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    #The argparse module makes it easy to write user-friendly command-line interfaces
    #Add required  groups
    #Create the arguments
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)") # Default value can be changed as per project requirement
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client = mqtt.Client()
    #Note that mqtt in the above was our imported alias
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client
 
def draw_boxes(frame,coord,prob_threshold, width, height):
    """
    Draw bounding boxes onto the frame
    :param frame: frame from camera/video
    :param result: list contains the data to parse draww_boxes
    :return: 1) person count and 2) frame
    """
    # Rectangle need two coordinates, one is top left corner and second one is bottom right
    # Top left corner will be (xmin,ymin)
    # Bottom right corner will be (xmax,ymax)
    
    # Set initial value i.e. counter, two points/coordinates for rectangle
    counter=0 
    initial_point=0 # Represent top left corner of rectangle
    ending_point=0 # Represent bottom right corner of rectangle
    # Loop through detections and determine what and where the objects are in the image
    # For each detection , it has 7 values i.e. [image_id,label,conf,x_min,y_min,x_max,y_max]
    for obj in coord[0][0]: 
        # In order to draw bounding box for object when it's probability is more than the specified threshold
        if obj[2] > prob_threshold: # Extract the confidence and compare with threshold value
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            initial_point = (xmin,ymin)
            ending_point = (xmax,ymax)
            # Use cv2.rectangle() method to draw a rectangle around detection 
            # Draw a rectangle with colored line (can be changed as per requirement) borders of thickness of 1 px
            # cv2. rectangle(img, pt1, pt2, color, thickness)
            frame = cv2.rectangle(frame, initial_point, ending_point, (250,0,50),1)
            counter+=1 # It will increase counter to one up as any detection detected
    return frame, counter
    # We return frame and counter from the above function which will help us later in counting detections
    # In short, The above code will draw a bounding box on actual input frame


    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by the function `build_argparser()`
    :param client= MQTT client
    :return will be None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    # Define a varaible for args.model
    model = args.model
    # Define a variable for args.device
    DEVICE = args.device
    # Define a variable for args.cpu_extension
    CPU_EXTENSION = args.cpu_extension
    # Loading the model through infer_network with our earlier defined variables i.e. model, CPU_EXTENSION & DEVICE 
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    # Get input shape from get_input_shape() through infer_network
    net_shape = infer_network.get_input_shape()
    # Handle the input stream i.e Handle the video, webcam or image
    Path_file = args.input
    # Flag for the input image and set it as False initially
    single_image_form = False
    # Checks for live feed i.e from any camera like webcam or other cams
    if Path_file == 'CAM':
        input_validated = 0

    # Checks for input image
    elif Path_file.endswith('.jpg') or Path_file.endswith('.bmp') :
        single_image_form = True
        # If there is an image input , set the single_image_form as True
        input_validated = Path_file

    # Checks for video file (This project is also focus on video file)
    else:
        input_validated = Path_file
        assert os.path.isfile(Path_file), "file doesn't exist"
        
    # Handle the input stream 
    #  In order to get and open video capture, we use follwing code
    capture = cv2.VideoCapture(Path_file)
    capture.open(Path_file)
    # Get the shape of input i.e. its height and width
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # In opencv we can replace CAP_PROP_FRAME_WIDTH with (3)
    # In the same maner we can replace CAP_PROP_FRAME_HEIGHT with (4)
    
    

    #iniatilizeing few variables 
    report = 0
    counter = 0
    prev_counter = 0
    prev_duration = 0
    total_counter = 0
    dur = 0
    request_id=0
    # Extracting size of image net_shape i.e. width and height
    w= net_shape[3] # Width of image
    h= net_shape[2] # Height of image
    # Loop/process the frame until stream/video is over
    # For this purpose , will use While Loop
    while capture.isOpened():
        # Read from the video capture/ frame
        flag, frame = capture.read()
        if not flag:
            break
        # Call a function for waitKey() as it waits for a key event for a "delay" here 60 miliseconds
        key_pressed = cv2.waitKey(60)
        

        # Pre-process the image as needed 
        """Given an input image (here it is "frame"), height(here it is "h") and width(here it is "h"):
                Resize to height and width
                Transpose the final "channel" dimension to be first
                Reshape the image to add a "batch" of 1 at the start""" 
        image = cv2.resize(frame, (w, h))
        pp_image = image.transpose((2, 0, 1))
        pp_image = pp_image.reshape(1, *pp_image.shape)
        #Start asynchronous inference for specified request
        #Perform inference on the frame
        duration_report = None
        inf_start = time.time()
        infer_network.exec_net(pp_image)
        
        # Get the output of inference
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start
            # Results of the output layer of the network
            output_results = infer_network.get_output()
            #Extract any desired stats from the results 
            #Update the frame to include detected bounding boxes
            box_frame, pointer = draw_boxes(frame,output_results,prob_threshold, width, height)
            # As we can extract two values from draw_boxes function i.e. frame and counter
            
            # Display the inference time
            message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(box_frame, message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 10, 10), 1)
            
            # Here we will write code to find number of counts and total counts
            # Now calculate and send relevant information on current counts, total counts and duration to MQTT server 
            # Topics are:
            # 1) "person": keys of "count" and "total"
            # 2) "person/duration": key of "duration" 
            if pointer != counter:
                prev_counter = counter
                counter = pointer
                if dur >= 3:
                    prev_duration = dur
                    dur = 0
                else:
                    dur = prev_duration + dur
                    prev_duration = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > prev_counter:
                        total_counter += counter - prev_counter
                    elif dur == 3 and counter < prev_counter:
                        duration_report = int((prev_duration / 10.0) * 1000)
            # The final piece for MQTT is to actually publish the statistics to the connected client            
            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': total_counter}),
                           qos=0, retain=False)
            # whereas: 
            # topic - the topic to publish to 
            # payload - the message to publish 
            # retain - whether the message should be retained
            if duration_report is not None:
                #The final piece for MQTT is to actually publish the statistics to the connected client
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
            
            #Send frame to the ffmpeg server
            #Once the output frame has been processed (drawing bounding boxes, semantic masks, etc.), 
            # you can write the frame to the stdout buffer and flush it
            sys.stdout.buffer.write(box_frame)
            sys.stdout.flush()
            
            #Save the image
            if single_image_form:
                cv2.imwrite('output_image.jpg', box_frame)

        # Break if escape key is pressed(27 is for esc key)
        if key_pressed == 27:
            break
        

    # Release the capture and destroy all windows of Opencv
    capture.release()
    cv2.destroyAllWindows()
    # At the end of processing the input stream, make sure to disconnect
    client.disconnect()
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

