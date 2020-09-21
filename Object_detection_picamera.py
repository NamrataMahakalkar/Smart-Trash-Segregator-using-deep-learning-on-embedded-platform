######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import math
import time
from adafruit_servokit import ServoKit

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    #Use smaller resolution for
#IM_HEIGHT = 480   #slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frz_infgrph_trs6cls.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','label_map_trs6cls.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

#Get XY coordinate of the detected object
def XY_cordiFun(i):
    #top gives distance of bbx from bottom of the image so to get the centre of the bbx (y1+y2)/2 where y1 is value of y  from the bottom and y2 is also value of y from the bottom and adjust the origin as 0.5
    Y_coordi=((boxes[0][i][1] + boxes[0][i][3])/2)-0.5
    #right gives distance of bbx from left of the image so to get the centre of the bbx (x1+x2)/2 where x1 is value of x  from the left and x2 is also value of x from the left and adjust the origin as 0.5
    X_coordi=((boxes[0][i][0] + boxes[0][i][2])/2)-0.5
    #scale by 72 at x and 52 at y as origin is shifted to centre of the screen divide the values by 0.5
    actual_x= X_coordi*(13/0.5)
    actual_y= Y_coordi*(11/0.5)
    coordi_xy= [actual_x, actual_y]
    return coordi_xy

#Calculate angle for motor 1 and motor 2 from x and y co-ordinate respectively
def angle_calculator(xy_values):
    thita1rad = math.atan(xy_values[1]/60)
    thita0rad = math.atan(xy_values[0]/60)
    thita1 = thita1rad*57.295 #radian to degree conversion
    thita0 = thita0rad*57.295 #radian to degree conversion
    #adding offset to the motor angles
    thita0 = -thita0 + 70
    thita1 = -thita1 + 70
    return [thita0,thita1]

#Servo motor angle increament and decreament function

kit = ServoKit(channels=16)

def slowRotationIncr(channelNo, xRotInDegree, CuRotInDegree):
    #if CuRotInDegree==0:
    #    kit.servo[channelNo].angle = 63
    #    print("Current angle at chennel "+str(channelNo)+" is "+str(CuRotInDegree)) 
    #else:
    rotInDegreeCon= ((CuRotInDegree-xRotInDegree)/0.2)
    curval= xRotInDegree
    for i in range (int(rotInDegreeCon)):
        curval=curval+0.2
        kit.servo[channelNo].angle = curval
        print("Current angle at chennel "+str(channelNo)+" is "+str(curval))
        time.sleep(0.01)

def slowRotationDncr(channelNo, xRotInDegree, CuRotInDegree):
    currAngleCon= ((xRotInDegree-CuRotInDegree)/0.2)
    curval= xRotInDegree
    for i in range (int(currAngleCon)):
        curval=curval-0.2
        kit.servo[channelNo].angle = curval
        print("Current angle at chennel "+str(channelNo)+" is "+str(curval))
        time.sleep(0.01)

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# declaire global variable to preserve previous rotation angle (in degree) of servo motor
xy_values_priv=[70, 70]
motor_angle_priv =[70, 70]

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame = frame[10:400,400:940]
        #print(frame.shape)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        #set the motor at origin
        kit.servo[0].angle = 70
        kit.servo[1].angle = 70
        #get the get [top, left, bottom, right] co-ordinate of the detected classes with >40 % probability
        #call function witch Calculate XY coordinate
        #call function which returns motor angles
        #print(num)
        detcount=0
        for i in range (0, 100) :
            if scores[0][i]> 0.4 :
                print("i="+ str(i))
                print("box co-ordnates are:") 
                print(boxes[0][i])
                xy_values = XY_cordiFun(i)
                motor_angle= angle_calculator(xy_values)
                print("Current XY values are: "+str(xy_values))
                print("Current Motor angle are: "+str(motor_angle))
                print("Privious XY values were: "+str(xy_values_priv))
                print("Privious Motor angle were: "+str(motor_angle_priv))
                kit.servo[1].angle = motor_angle[0]
                kit.servo[0].angle = motor_angle[1]
               # if motor_angle_priv[0]<motor_angle[0] and motor_angle_priv[1]<motor_angle[1]:
                #   slowRotationIncr(0, motor_angle_priv[0], motor_angle[0])
                 #   slowRotationIncr(1, motor_angle_priv[1], motor_angle[1])
                #elif motor_angle_priv[0]<motor_angle[0] and motor_angle_priv[1]>motor_angle[1]:
                 #   slowRotationIncr(0, motor_angle_priv[0], motor_angle[0])
                  #  slowRotationDncr(1, motor_angle_priv[1], motor_angle[1])
                #elif motor_angle_priv[0]>motor_angle[0] and motor_angle_priv[1]<motor_angle[1]:
                 #   slowRotationDncr(0, motor_angle_priv[0], motor_angle[0])
                  #  slowRotationIncr(1, motor_angle_priv[1], motor_angle[1])
                #elif motor_angle_priv[0]>motor_angle[0] and motor_angle_priv[1]>motor_angle[1]:
                 #   slowRotationDncr(0, motor_angle_priv[0], motor_angle[0])
                  #  slowRotationDncr(1, motor_angle_priv[1], motor_angle[1])
                detcount=detcount+1
                print("class is "+str(classes[0][i]))
                #time.sleep(5)
                xy_values_priv=xy_values
                motor_angle_priv = motor_angle
        print("Number of detections were "+ str(detcount))
        print("***********************************************************************")
 
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
cv2.destroyAllWindows()


