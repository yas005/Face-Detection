import cv2
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--path', help="no plot is shown.")
args = parser.parse_args()

os.makedirs("output_frame")

vidcap = cv2.VideoCapture(args.path)
success,image = vidcap.read()
count = 0
while success:

  cv2.imwrite("./output_frame/%06d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

print("finish! convert video to frame")
