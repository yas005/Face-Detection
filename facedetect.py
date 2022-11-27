from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob
from natsort import natsorted
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree
from tqdm import tqdm
import argparse
import time
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets.voc_classes import VOC_CLASSES
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np
import torch
import argparse

#<추가---도커에 추가해야함>
import av     # PyAV 임포트
from PIL import Image     # Pillow 임포트



#yolox code start

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]




def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    # 사용할 모델 (deepyolox or yolox)
    parser.add_argument("-n", "--name", type=str, default='yolox-x', help="model name")
    # 변경---------------------------
    # 비디오를 이미지로 만들때 생성되는 이미지 저장하는 곳
    parser.add_argument('-p', "--path", type=str, default='./test_dataset/test_image', help="choose a video or image path")
    # facedetect path
    parser.add_argument('-s', "--pred_save_path", type=str, default='./test_dataset/yolox_deepsort_pred2/', help="path for saving prediction xml")
    # facedetect_img path
    parser.add_argument('-i', "--pred_save_img", type=str, default='./test_dataset/test_pred_image2/', help="path for saving prediction image")
    # 원본 비디오 path ( 원본비디오 있는 경로 )
    parser.add_argument('-vp',"--test_path", type=str, default=None, help="test dataset path(video or image)")
    # pred 값이 저장되는 경로 지정(pred 이미지,annotation)
    parser.add_argument('-sp',"--save_path",type=str,default="./test_dataset/",help='path for saving prediction')
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    #width, height
    parser.add_argument("--width",type=int, default=224,help="frame width" )
    parser.add_argument("--height",type=int, default=224,help="frame height" )
    #backbone 선택(x.l,m,s)
    parser.add_argument("--back",type=str,default='x', help="choose yolox version")
    #pred이미지 저장유무 선택 (y:yes, n:no)
    parser.add_argument(
        "--save",
        type=str,
        default = "y",
        help="whether to save the inference result of image/video"
    )
    parser.add_argument(
        "--annotation_path", default="./assets/", help="path to images or video"
    )
    #비디오_info 저장 경로
    parser.add_argument(
        "--information_path", default="./assets/", help="path to images or video"
    )
    #삭제가능
    parser.add_argument(
        "--image_data_path", default="./test_dataset/test_image", help="path to images or video"
    )
    #삭제가능
    parser.add_argument(
        "--res_save_path", default="res", help="path to images or video"
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    #학습된 yolox 가중치 선택
    parser.add_argument("-c", "--ckpt", default='./yolox_x.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser



#name change
def txt_maker_yolox(args, bboxes, img_info, scores):
    f = open(args.annotation_path, 'w')
    bbox = np.array(bboxes).astype(np.int32)
    scores = scores.detach().cpu().numpy()
    i = 0
    # print(scores)
    for item in bbox:
        f.write('head' + ' ' + str(scores[i]) +' '+ str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
        i += 1
    f.close()

def video2image():
        video_name = args.test_path.split("/")[-1].split(".")[0]
        # 원본 비디오 정보 저장 파일
        args.information_path = video_name + "_info.txt"
        time_name = time.strftime("%Y%m%d%H%M", time.localtime())
        #자른 이미지 넣는 폴더 생성 
        os.makedirs(args.save_path+"/test_video/"+args.name+"/"+video_name +"_"+time_name+"/")
  
        vidcap = cv2.VideoCapture(args.test_path)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(args.save_path+"/test_video/"+args.name+"/"+video_name +"_"+time_name+"/"+video_name +"_"+ "%06d.jpg" % count, image)
            success,image = vidcap.read()
            count += 1
            
        return time_name

def img2video(img_path,video_path,fps):
    
    img_list = os.listdir(img_path)
    img_list = natsorted(img_list)
    img_list = [os.path.join(img_path, img_name) for img_name in img_list]
    img = cv2.imread(img_list[0])
    size = (img.shape[1], img.shape[0])
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    for img_name in img_list:
        img = cv2.imread(img_name)
        video.write(img)
    video.release()


def video_info(time_name,ver):

        video = args.test_path.split("/")[-1]
        video_name = video.split(".")[0]
    # 동영상을 임포트
        container = av.open(args.test_path)
        video = container.streams.video[0]

        frames = container.decode(video=0)

        # 프레임 단위로 동영상에서 이미지 추출 
        for frame in frames:
            frame.to_image().save('frame-%04d.jpg' % frame.index)
            # 이미지 하나 추출 뒤 포문 탈출
            break

        # 동영상 재생시간
        time_base = video.time_base
        # fps 샘플링 추출
        fps = video.average_rate
    
        movie = int(int(video.frames)/int(fps))
        sec = movie %60
        minute = movie // 60 
        
        #변경사항(문제) : video 의 재생 시간이 분수로 나옴
        video_info=[minute,sec,args.width, args.height,fps]
        
        f = open(args.save_path+"/"+ver+"/"+args.name+"/"+video_name+"_"+time_name+"/"+ args.information_path, 'w')
        
        f.write('영상 길이 :' + ' ' + str(video_info[0])+'m' +str(video_info[1])+'s' +'\n'+ '프레임 너비, 높이 : '+ str(video_info[2]) + ' ,' + str(video_info[3]) + '\n'+'프레임 속도 : ' + str(video_info[4]) +'\n')
        f.close()

        return fps



#deepsort code start

#deeepsort class
class FaceDetection_T():
    def __init__(self,path,resize,test_path,name,save_path,information_path,pred_save_path,pred_save_img):
        self.path = path
        self.resize = resize
        self.test_path = test_path
        self.name = name
        self.save_path = save_path
        self.information_path = information_path
        self.pred_save_path = pred_save_path
        self.pred_save_img  = pred_save_img

    # annotation파일 만드는 함수 
    def txt_maker_deepsort(self,bboxes, img_info):
        f = open(os.path.join(self.pred_save_path, os.path.basename(img_info['file_name']).rsplit('.')[0]+'.txt'), 'w')
        bbox = np.array(bboxes)
        for item in bbox:
            f.write('head' +' '+str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
        f.close()

    def track_images(self, args):
        version = args.name.split("-")[-1]
        model_version = 'yolox-' + version
        
        tracker = Tracker(model=str(model_version), ckpt=args.ckpt,filter_class=['face'])
        imgs = natsorted(glob(os.path.join(self.path,'*.png')) + glob(os.path.join(self.path,'*.jpg')) + glob(os.path.join(self.path,'*.jpeg')) + glob(os.path.join(self.path,'*bmp')))
        
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
            
            args.width = width
            args.height = height

            im = cv2.imread(path)
            #resize 옵션 변경(이미지 frame 크기)
            im = imutils.resize(im, height=self.resize)
            test_size = im.shape
            ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
            img_info["ratio"] = ratio
            image, output = tracker.update(im)
            if len(output) !=0:
                output= output.astype('float64')
                for i in range(len(output)):
                    output[i][:-1] /= ratio
                output = output.astype('int64')
           
            self.txt_maker_deepsort(output, img_info)

            cv2.imwrite(os.path.join(self.pred_save_img, os.path.basename(path)), image)

    

    def track_cap(self,file):
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
            im = imutils.resize(im, height=self.resize)
            image,output = tracker.update(im)
            print(output)
            cv2.imshow('demo', image)
            cv2.waitKey(1)
            if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

    #deepsort 돌아가는 함수
    def demo_tracking(self):
        
        
        os.makedirs(self.pred_save_path, exist_ok=True)
        os.makedirs(self.pred_save_img, exist_ok=True)

        if os.path.isfile(self.path):
            self.track_cap(self.path)
        else:
            
            self.track_images(args)
        
        
        
    

  
    
   
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

#name change
def txt_maker_yolox(args, bboxes, img_info, scores):
    f = open(args.annotation_path, 'w')
    bbox = np.array(bboxes).astype(np.int32)
    scores = scores.detach().cpu().numpy()
    i = 0
    # print(scores)
    for item in bbox:
        f.write('head' + ' ' + str(scores[i]) +' '+ str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
        i += 1
    f.close()




def image_demo(predictor, vis_folder, path, current_time,save_folder):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info, width, height = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

        if (args.save == "y"):
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)


        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    return width, height


def yolox_main(exp, args, pred_save_img):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    

    if args.trt:
        args.device = "gpu"

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
        

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        
        ckpt = torch.load(ckpt_file, map_location="cpu")
   
        model.load_state_dict(ckpt["model"], strict =False)
     

    if args.fuse:
 
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs

    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, VOC_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if (args.demo == "video" or args.demo == "image"):
        width, height = image_demo(predictor, vis_folder, args.path, current_time,pred_save_img)

    args.width = width
    args.height = height
    

def demo_yolox(path,pred_save_path,pred_save_img):
    exp = get_exp(args.exp_file, args.name)

    dir, file = os.path.split(path)
    os.makedirs(pred_save_path, exist_ok=True)
  
    
    
    for file in tqdm(sorted(os.listdir(path))):
        format = str(file.split(".")[-1])
        args.path = os.path.join(path, file)
        args.annotation_path = os.path.join(pred_save_path, file.replace(format,'txt'))
        yolox_main(exp, args,pred_save_img)
    
    


#yolox 예측 클래스
class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = (args.tsize,args.tsize)
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info, width, height

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        if output is None:
            f = open(args.annotation_path, 'w')
            f.close()
            return img

        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        if len(bboxes) !=0:
            txt_maker_yolox(args, bboxes, img_info, scores)

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res





if __name__ == "__main__":
    
    args = make_parser().parse_args()
    video_name = args.test_path.split("/")[-1].split(".")[0] #비디오 이름
    time_name = video2image() #비디오를 이미지로 만들때, 생성 시간(폴더 이름)
    save_folder = video_name + "_" + time_name #저장 폴더 이름
    video_dir = os.path.join(args.save_path,"output_video/") #결과 비디오 저장 폴더

    if not os.path.exists(video_dir):
            os.makedirs(video_dir)
    
    video_name = save_folder+".mp4" #저장이름
    video_path = os.path.join(video_dir,video_name)


    if (args.demo == "video"): 
        args.path = os.path.join( args.save_path,"test_video",args.name,save_folder) #비디오에서 추출한 이미지 저장 경로


    else:
        args.path = args.test_path

    #deepyolox인지 yolox인지 구분
    ver = args.name.split("-")[0]

    if (ver == "deepyolox" ):
       
        if(args.save== "y"):
            args.pred_save_path = os.path.join(args.save_path,ver,args.name,save_folder,"facedetect/") 
            args.pred_save_img = os.path.join(args.save_path,ver,args.name,save_folder,"facedetect_img/")
           
        elif(args.save== "n"):
            args.pred_save_path = os.path.join(args.save_path,ver,args.name,save_folder,"facedetect/")
           
        
        # deepsort class사용
        # class 속성 : path,resize,video_path,name,save_path,model,information_path,pred_save_path,pred_save_img
        deepsort = FaceDetection_T(args.path,args.tsize,args.test_path,args.name,args.save_path, args.information_path,
        args.pred_save_path, args.pred_save_img)

        deepsort.demo_tracking()
        if(args.demo == "video"):
            fps =  video_info(time_name,ver)
            img2video(args.pred_save_img,video_path,int(fps))
     
    elif(ver == "yolox"):
        
        
        if(args.save== "y"):
            args.pred_save_path = os.path.join(args.save_path,ver,args.name,save_folder,"facedetect/") 
            args.pred_save_img = os.path.join(args.save_path,ver,args.name,save_folder,"facedetect_img/")
 
        elif(args.save== "n"):
            args.pred_save_path = os.path.join(args.save_path, ver,args.name,save_folder,"facedetect/")
    


        demo_yolox(args.path,args.pred_save_path,args.pred_save_img)

        if(args.demo == "video"):
            fps = video_info(time_name,ver)
            img2video(args.pred_save_img,video_path,int(fps))
