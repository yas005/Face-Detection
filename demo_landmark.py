from tkinter import Y
from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob
from natsort import natsorted
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree
from tqdm import tqdm

import sys

import numpy as np
import pickle
import importlib
from math import floor
import time


import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, './lib')

from networks import *
import data_utils
from functions import *
from mobilenetv3 import mobilenetv3_large

from experiments.data_300W.pip_32_16_60_r101_l2_l1_10_1_nb10 import Config



#device

        

def xml_maker(args, bboxes, img_info):
    bbox = np.array(bboxes)
    filename = img_info['file_name']
    width = img_info['width']
    height = img_info['height']
    object_len = len(bbox)

    label = 'head'
    
    root = Element('annotation')
    SubElement(root, 'folder').text = 'HollywoodHeads'
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = str(os.path.split(args.path)[1])
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'HollywoodHeads 2015 Database'
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '3'
    
    SubElement(root, 'segmented').text = '0'

    for item in bbox:
        label = label
        xmin = item[0]
        ymin = item[1]
        xmax = item[2]
        ymax = item[3]

        object = SubElement(root, 'object')
        SubElement(object, 'name').text = label
        SubElement(object, 'pose').text = 'Unspecified'
        SubElement(object, 'truncated').text = '0'
        SubElement(object, 'difficult').text = '0'
        
        bndbox = SubElement(object, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(xmin)
        SubElement(bndbox, 'ymin').text = str(ymin)
        SubElement(bndbox, 'xmax').text = str(xmax)
        SubElement(bndbox, 'ymax').text = str(ymax)
          
    tree = ElementTree(root)

    return tree

def txt_maker(args, bboxes, img_info):
    f = open(os.path.join(args.pred_save_path, os.path.basename(img_info['file_name']).rsplit('.')[0]+'.txt'), 'w')
    bbox = np.array(bboxes)
    for item in bbox:
        f.write('head' + ' ' + '0.7'+' '+str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
    f.close()

#랜드마크 생성 함수
def track_lmks(image,output):
    input_size =256
    net_stride = 32
    num_nb = 10
    
        
    if len(output) != 0:
        for item in output:
            x1 = item[0]
            y1 =item[1]
            
            x2 = item[2]
            y2 = item[3]

            image_height = image.shape[0]
            image_width = image.shape[1]

            det_xmin = min(x1,x2)
            det_xmax = max(x1,x2)
            det_ymin = min(y1,y2)                
            det_ymax = max(y1,y2)

            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width-1)
            det_ymax = min(det_ymax, image_height-1)

            det_width = det_xmax - det_xmin
            det_height = det_ymax - det_ymin

              
               
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            det_crop = cv2.resize(det_crop, (input_size, input_size))
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()
            for i in range(cfg.num_lms):
                x_pred = lms_pred_merge[i*2] * det_width
                y_pred = lms_pred_merge[i*2+1] * det_height
                cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
    
            


    



def track_images(args):
    tracker = Tracker(model='yolox-x', ckpt='yolox_x.pth',filter_class=['face'])
    imgs = natsorted(glob(os.path.join(args.path,'*.png')) + glob(os.path.join(args.path,'*.jpg')) + glob(os.path.join(args.path,'*.jpeg')))
    #change
    my_thresh = 0.9
    det_box_scale = 1.0

    for path in tqdm(imgs):

        img_info = {"id": 0}
        if isinstance(path, str):
            img_info["file_name"] = os.path.basename(path)
            img = cv2.imread(path)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        

        frame = cv2.imread(path)

        
        frame = imutils.resize(frame, height=400)
        test_size = frame.shape
        ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        image, output = tracker.update(img)
        
        #랜드마크 함수
        track_lmks(image,output)
        
        
        # if len(output) !=0:
        #     output= output.astype('float64')
        #     for i in range(len(output)):
        #         output[i][:-1] /= ratio
        #     output = output.astype('int64')     
            
            
        txt_maker(args, output, img_info)

   

        cv2.imwrite(os.path.join(args.pred_save_img, os.path.basename(path)), image)




def track_cap(file):
    cap = cv2.VideoCapture(file)
    tracker = Tracker()
    a = 0
    while True:
        
        _, im = cap.read()
        if im is None:
            break
        a += 1
        if a%10!=0:
            continue
        im = imutils.resize(im, height=640)
        image,output = tracker.update(im)
        print(output)
        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker-landmark Demo!")
    parser.add_argument('-p', "--path", type=str, default='./test_dataset/test_image', help="choose a video or image path")
    parser.add_argument('-s', "--pred_save_path", type=str, default='./test_dataset/yolox_deepsort_lmk_pred/', help="path for saving prediction xml")
    parser.add_argument('-i', "--pred_save_img", type=str, default='./test_dataset/test_pred_lmk_image/', help="path for saving prediction image")
    parser.add_argument('-e',"--experiments",type=str,default='./experiments/data_300W/pip_32_16_60_r101_l2_l1_10_1_nb10.py')
    args = parser.parse_args()
    config_path = args.experiments

    
    cfg = Config()
    cfg.experiment_name = config_path.split('/')[-1]
    cfg.data_name = "data_300W"

    
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)

    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

    os.makedirs(args.pred_save_path, exist_ok=True)
    os.makedirs(args.pred_save_img, exist_ok=True)

    if os.path.isfile(args.path):
        track_cap(args.path)
    else:
        track_images(args)
        
