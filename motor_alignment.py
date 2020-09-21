from adafruit_servokit import ServoKit
from picamera import PiCamera
kit = ServoKit(channels=16)
import cv2
import keyboard, sys, tty, os, time, termios

def getch():
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(sys.stdin.fileno())
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	return ch

kit.servo[0].angle = 0
kit.servo[1].angle = 0
x_angle = 0
y_angle = 0
camera = PiCamera()
#camera.resolution = (640, 480)
camera.start_preview()
while (1):
	char = getch()
	if (char == 'q'):
		camera.stop_preview()
		break
	if (char == 'w'):
		y_angle = y_angle + 1
		if y_angle in range(0, 180):
			kit.servo[1].angle = y_angle
		else:
			print("y_angle out of range")
	if (char == 's'):
		y_angle = y_angle - 1
		if y_angle in range(0, 180):
			kit.servo[1].angle = y_angle
		else:
			print("y_angle out of range")
	if (char == 'a'):
		x_angle = x_angle + 1
		if x_angle in range(0, 180):
			kit.servo[0].angle = x_angle
		else:
			print("x_angle out of range")
	if (char == 'd'):
		x_angle = x_angle - 1
		if x_angle in range(0, 180):
			kit.servo[0].angle = x_angle
		else:
			print("x_angle out of range")
	time.sleep(0.1)
	print ("the x_angle of this point is")
	print (x_angle)
	print ("the y_angle of this point is")
	print (y_angle)
