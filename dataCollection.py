from picamera import PiCamera
import os
import sys, tty, termios
import keyboard
from scipy.misc import imread
from picamera import PiCamera
import time
camera = PiCamera()

def getch():
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(sys.stdin.fileno())
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	return ch

print("we are about to read a file")
file1= open("/home/pi/dataset/count.txt","r")
print("count file is read")
count = file1.readline()
count_int = int(count)
print (count)
file1.close()

while True:
	count_int = count_int+1
	file2= open("/home/pi/dataset/count.txt","w")
	count_str = str(count_int)
	file2.write(count_str)
	file2.close()
	print('press y to continue and x to break the loop')
	filename="/home/pi/dataset/trash"+str(count_int)+str(".JPG")
	char = getch()
	if(char == "y"):
		camera.start_preview()
		time.sleep(0.5)
		camera.capture(filename)
		time.sleep(0.5)
		camera.stop_preview()
		# The "x" key will break the loop and exit the program
	if(char == "x"):
		print("Program Ended")
		break	


