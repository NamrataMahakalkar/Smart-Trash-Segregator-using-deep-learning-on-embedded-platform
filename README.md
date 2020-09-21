# Smart-Trash-Segregator-using-deep-learning-on-embedded-platform
Here is a compete project of trash segregation for the indian trash which uses deep learning. The dataset for the same project is available at https://drive.google.com/drive/folders/15Y-9jHfXLhF7rTLaiC8RHoSqOjoSDSZ2?usp=sharing. The system code is novel starts from the datacollection on Raspberry-Pi till the system actuations. The code of the object detection model is not completely new, rather here I have joined the bits and pisces of existingly available code with moderate level of modifications for my problem statement and dataset. There are many models in the system. 

1) Modules required for the actual system are dscribed bellow in the format such as [Module name,Functionality,Code type,No. of lines of code]
dataCollection.py	For collecting data from the pi-camera and naming it appropriately in sequence irrespective of Pi restarts. -Python
in Raspberry pi	50
Colab_objectDetection_training.py	Model is trained for the custom dataset using SSD_MobileNet_v2 model on Google Colab. Following are the steps:	Python
in Google colab	212
	1)	Mount the drive		2
	2)	Install Tensorflow 1.15		1
	3)	Navigate to the data folder and check the folder hierarchy 		2
	4)	Make test and train directory		2
	5)	Divide the data in 80:20 proportion in train and test folder respectively		4
	6)	Installation of protobuf -compiler and matplotlib		2
	7)	Import the necessary file		18
	8)	Navigate to data folder and create csv from xml		38
	9)	Download tensorflow pre-trained models in the correct directory		2
	10)	 Compilation of protobuf		4
	11)	 Testing model builder		2
	12)	 Specify the classes for the custom object detection and create record file using data.		72
	13)	Specify the tensorflow model to train for custom data		24
	14)	Open config file to specify training parameters.		1
	15)	Edit config file with the required changes		6
	16)	Initialize tensor board for the performance check  and create link for the visualization		12
	17)	 Finally train the model		3
	18)	 Download trained model and label map files		17
ImageAi_Object_detection.py	Steps:
1)	Mount the drive
2)	Install tensorflow 1.13
3)	Install ImageAi
4)	Train the model
5)	Visualize the performance	Python  in Google colab	19
Motor_alignment.py	After installation of trained model, origin needs to set at the centre of trash platform.
After motor alignment centre will be considerate as (0,0) for the further calculation of motor angles.	Python in Google colab	100
Object_detection_picamera.py	Object detection in raspberry pi using trained object detection model.
•	Continuous capturing and detecting an object.
•	Scaling unit for computing actual XY co-ordinates in the trash platform field.
•	Inverse kinematics for angle calculation.
•	Servo motor rotation management	Python
in Raspberry pi	269
2) Modules required for the Simulation setup
Colab_objectDetection_training_infrc_computation.py	This script comprise of functionality of Colab_objectDetection_training.py till step 18. Following task are performed in order to accomplish the simulation task.
•	Taking out inference over captured and uploaded image by mobile phone to the G-drive.
•	Apply scaling computation.
•	Apply inverse kinematics and compute the x and y motor angles of robotic arm.
•	Write x and y angle as well as the index of detected object to the text file in x---y---c- format. E.g. x070y080c1	Python in Google colab	394
Ser_xy_to_servo_vac_on_off.ino	•	Initialize and setup servoX , servoY and vacuum pump pin numbers in arduino.
•	Setup serial communication.
•	Write functions for the slow rotation of servo motor in forward abd backward direction for both the motors.
•	Write function to check the serial port 10 byte string and parse it. Provide XY angle to the XY motors, turn on the vacuum pump, wait for 5 sec, from class index select XY motor angle from switch case and take the action, turn off the vacuum pump and wait for next angles. 
•	Call the function in loop.	C in Arduino IDE	178
