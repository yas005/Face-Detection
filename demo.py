from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob
from natsort import natsorted
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree
from tqdm import tqdm


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


def track_images(args):
    tracker = Tracker(model='yolox-x', ckpt='yolox_x.pth',filter_class=['face'])
    imgs = natsorted(glob(os.path.join(args.path,'*.png')) + glob(os.path.join(args.path,'*.jpg')) + glob(os.path.join(args.path,'*.jpeg')))

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
        

        im = cv2.imread(path)
        im = imutils.resize(im, height=400)
        test_size = im.shape
        ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        image, output = tracker.update(im)
        if len(output) !=0:
            output= output.astype('float64')
            for i in range(len(output)):
                output[i][:-1] /= ratio
            output = output.astype('int64')
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
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, default='./test_dataset/test_image', help="choose a video or image path")
    parser.add_argument('-s', "--pred_save_path", type=str, default='./test_dataset/yolox_deepsort_pred/', help="path for saving prediction xml")
    parser.add_argument('-i', "--pred_save_img", type=str, default='./test_dataset/test_pred_image/', help="path for saving prediction image")
    args = parser.parse_args()
    
    os.makedirs(args.pred_save_path, exist_ok=True)
    os.makedirs(args.pred_save_img, exist_ok=True)

    if os.path.isfile(args.path):
        track_cap(args.path)
    else:
        track_images(args)
        