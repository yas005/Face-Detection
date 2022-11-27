import cv2
import numpy as np
import os
import sys
import argparse
import natsort

parser = argparse.ArgumentParser()
parser.add_argument( '--video_name', help="no plot is shown.",default="result")
args = parser.parse_args()

#image to video converter
def img2video(img_path, video_path, fps=23):
    img_list = os.listdir(img_path)
    img_list = natsort.natsorted(img_list)
    img_list = [os.path.join(img_path, img_name) for img_name in img_list]
    img = cv2.imread(img_list[0])
    size = (img.shape[1], img.shape[0])
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    for img_name in img_list:
        img = cv2.imread(img_name)
        video.write(img)
    video.release()

if __name__ == '__main__':

    name = str(args.video_name) + ".mp4"
    path = os.path.join("./",name)
    img2video("./input", path)
    
    
    
