# Face Detection and Tracking (YOLOX, YOLOX + DeepSORT)

![ezgif com-gif-maker](https://user-images.githubusercontent.com/44921488/192144865-cf819b6b-c066-417f-8662-35c59fce4677.gif)

##### ** meaning of the number at the top of the bounding box = confidence score
( 관련 설명 : https://techblog-history-younghunjo1.tistory.com/178 )

## 1. Get Ready
dataset download : [HollywoodHeads Dataset](https://drive.google.com/file/d/1T5LWSezp2xeSqr1GBskOtBczXQ8ew6q7/view?usp=sharing)  
Wider-face trained YOLOX-X Weight download : [yolox_x.pth](https://drive.google.com/file/d/17U4TgZf7crBV8yZ1kGHt0MeU6hLn2Vfl/view?usp=sharing)    
(place ```"yolox_x.pth"``` at ```"./yolox_x.pth"```)   
Wider-face trained YOLOX-L Weight download : [yolox_l.pth](https://drive.google.com/file/d/1R7G5HaLwRjoRgiFU9SN8fKWgoN1sMHro/view?usp=sharing)  
(place ```"yolox_l.pth"``` at ```"./yolox_l.pth"```)  
Wider-face trained YOLOX-M Weight download : [yolox_m.pth](https://drive.google.com/file/d/1pZrGu16NNXG7hZpxx3-k4ShvKa2qLbyt/view?usp=sharing)    
(place ```"yolox_m.pth"``` at ```"./yolox_m.pth"```)   
PIPNet Weight download : [epoch59.pth](https://drive.google.com/file/d/1xmdZogBqFH9n55gBnpBjZvcG0QBMzFIH/view?usp=sharing)      
(place ```"epoch59.pth"``` at ```"./YOLOX/epoch59.pth"```)
##### If you want to test with your video dataset, you have to change the video to images.
```
python video2image.py --path "video path"
```
When change is done, you can get images at a directory (```"./output_frame".```)

For test, move all images in a directory(```"./output_frame"```) to directory(```"./test_dataset/test_image"```)



## 2. Installation
```
conda create -n yolox python=3.8
conda activate yolox
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install scipy opencv_python_headless numpy pandas imutils matplotlib pycocotools thop tqdm easydict tabulate loguru apex ipdb megengine motmetrics onnx onnxruntime openvino pyyaml seaborn tensorrt natsort tqdm ninja
```
```
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

#### update requirements.txt (2022_11_14)
```
Python library
- 유지 : 1~15
- 변경 : 16~17 (downgrade)
- 추가 : 18~24
```

- #### pytorch version(2022_11_14)     
cvlab : 1.11.0  
dgcab(Dockerfile) : 1.8.2

As a result of comparising the two versions, the inference time is 3times different.    
(1.11.0 : 13 minutes, 1.8.2 : 46 minutes on a deepyolox-m model)    
```It is recommended to use a higher version of PyTorch.```



## 3. YOLOX-X Evaluation on HollywoodHeads Test Dataset
```
cd YOLOX
python tools/demo.py image -n yolox-x -c "../yolox_x.pth" --conf 0.25 --nms 0.45 --tsize 640 --device gpu --image_data_path "../test_dataset/test_image"
```
to save the inference result of image, add ```--save result``` at the end of the command line.  
When inference is done, you can get a predicted annotation at ```"./test_dataset/yolox_x_pred".```  
For evaluation, move ```"./test_dataset/yolox_x_pred"``` folder to ```"./mAP/input/yolox_x_pred"```

```
cd ..
cd mAP
python main.py --pred_path yolox_x_pred --gt_path test_annotations_txt
```
#### YOLOX-X mAP : 70.33% (Train: Wider Face, Test: HollywoodHeads)

## 4. YOLOX-X + DeepSORT Evaluation on HollywoodHeads Test Dataset
```
python demo.py --path test_dataset/test_image --pred_save_path ./test_dataset/yolox_deepsort_pred/ --pred_save_img ./test_dataset/test_pred_image/
```
you can get a predicted annotation at ```"./test_dataset/yolox_deepsort_pred".```  
move ```"./test_dataset/yolox_deepsort_pred"``` to ```"./mAP/input/yolox_deepsort_pred"```

```
cd mAP
python main.py --pred_path yolox_deepsort_pred --gt_path test_annotations_txt
```
#### YOLOX-X + DeepSORT mAP : 71.91% (Train: Wider Face, Test: HollywoodHeads)

If you want to use image files in ```bitmap format``` , you have to change the code``` "def track_images()"``` as below.                     
[before]     
``` 
imgs = natsorted(glob(os.path.join(args.path,'*.png')) + glob(os.path.join(args.path,'*.jpg')) + glob(os.path.join(args.path,'*.jpeg')))
```
[after]
```
imgs = natsorted(glob(os.path.join(args.path,'*.png')) + glob(os.path.join(args.path,'*.jpg')) + glob(os.path.join(args.path,'*.jpeg'))+ glob(os.path.join(args.path,'*.bmp')))
```


## 5. YOLOX-X + PIPNet landmark Test on HollywoodHeads Test Dataset
```
cd YOLOX
python tools/demo_landmark.py image -n yolox-x -c "../yolox_x.pth" --conf 0.25 --nms 0.45 --tsize 640 --device gpu --image_data_path "../test_dataset/test_image"
```
to save the inference result of image, add ```--save result``` at the end of the command line.  
When inference is done, you can get a predicted annotation at ```"./test_dataset/yolox_x_lmk_pred",```
and you can get a predicted landmark image at ```"./YOLOX_outputs/yolox_x/vis_res_lmk" ```


## 6. YOLOX-X + DeepSORT + PIPNet landmark Test on HollywoodHeads Test Dataset

```
python demo_landmark.py --path test_dataset/test_image --pred_save_path ./test_dataset/yolox_deepsort_lmk_pred/ --pred_save_img ./test_dataset/test_pred_lmk_image/
```
you can get a predicted annotation at ```"./test_dataset/yolox_deepsort_lmk_pred".```
and you can get a predicted landmark image at ```"./test_dataset/yolox_deepsort_lmk_image"```


##### [Options]![image (2)](https://user-images.githubusercontent.com/88639269/194553890-85035967-60a1-4d60-8448-a2958dc16ee4.png)

## 7. To check the results on video 
1. make a directory
```
cd video
mkdir input
```
2. move all images to a directory(```"./video/input"```)
3. ``` python image2video.py --video_name "video name"```

   ex) python image2video.py --video_name yolox_result (= create "yolox_result.mp4")


## 8. To train YOLOX-X on Custom Dataset (e.g., HollywoodsHead Dataset)
You need a dataset with COCO style data format.  
To convert Pascal VOC format to COCO format, please refer [voc2coco](https://github.com/yukkyo/voc2coco)  
Put your data at ```./YOLOX/datasets/{YOUR_DATASET}```
```
python tools/train.py -n yolox-x -d 1 -b 16 --fp16 --logger wandb wandb-project yolox-face
```

## 9. Resize Test on HollywoodHeads Test Dataset

Test Dataset : HollywoodHead Dataset      
Size : 224 x 528

[ Resize image -- mAP(%)     ]
```
1. 95 x 224 : 68.49%
2. 224 x 528 (original size) : 72.01%
3. 400 x 942 : 71.91%
```
#### Result:
If the image size is resized smaller than the original image, performance(mAP matrix value) will be lower as expected. 
However, the performance does not deteriorate much.    
If the image size is resized larger than the original image, it is similar to the original image performance.(be almost same) 

[ mov_001_030700.jpeg ]

![mov_001_030700_224](https://user-images.githubusercontent.com/88639269/197601789-95582461-4ccc-4507-92e4-6302e18d35ca.jpeg)
![mov_001_030700](https://user-images.githubusercontent.com/88639269/197601733-0b79ae36-7f9a-45a0-a481-10781ee946ca.jpeg)


## 10. 2022_11_10 update

#### Video dataset Download
[HOLLYWOOD.mp4](https://drive.google.com/file/d/1OCFJz6h26uqTBUca2hQDerNl1noOcN5z/view?usp=sharing)    
(place ```"HOLLYWOOD.mp4"``` at ```"./video/HOLLYWOOD.mp4"```)    
#### Image dataset Download   
[HOLLYWOOD.jpeg](https://drive.google.com/file/d/1IC3CTQxkKWzI2RZmwmi-BHO7J1GcRe2i/view?usp=sharing)     
(place ```"HOLLYWOOD.jpeg"``` at ```"./test_dataset/test_image/HOLLYWOOD"```)  

#### [options]

```
1. video / image : input이 영상인지, 한개의 이미지인지 선택
2. --tsize : frame(이미지) 크기 조절 (int)
3. --test_path : video 원본 영상 경로/ image 원본 경로
4. --name : 모델 선택 
5. --ckpt : widerface로 학습된 yolox의 가중치 경로
6. --save_path : face detection 이미지, 정보 저장 경로
7. --save : 얼굴탐색이미지 저장 여부(y/n)
```
#### How to use the options (base)

python facedetect.py  ```(video/image)```  --name ```"사용할 모델-backbone버전"``` --ckpt ```"모델에 맞는 가중치"```  --test_path   ```"원본영상/이미지 경로"``` 

--save_path ```"얼굴탐색 이미지/정보를 저장할 경로"``` --save ```"얼굴탐색 이미지 저장 여부(y/n)"```

#### --name
- yolox model : yolox-x, yolox-l, yolox-m
(3가지 모델 비교 : https://github.com/Megvii-BaseDetection/YOLOX) 
- yolox+deepsort model : deepyolox-x, deepyolox-l, deepyolox-m
- model and weight(--ckpt)

![image](https://user-images.githubusercontent.com/88639269/201102288-291469e9-8914-4365-b010-10a215ff5f4f.png)




- deepyolox-x = yolox-x + deepsort 
- deepyolox-l = yolox-l + deepsort 
- deepyolox-m = yolox-m + deepsort

#### --tsize (resize)

- yolox model : cannot resize image ```( cannot use --tsize )``` : fixed at 640  
- yolox+deepsort model : can resize image ```( can use --tsize )```   
- code : ```im = imutils.resize(im, height=self.resize)```    
Q. Why do you only get a height value?  
A : When the height value is received, the size is changed based on the width and height proportional values of the original frame.     
ex) ```original frame size : 400 x 200```     &rarr;     ```--tsize 300```     &rarr;    ```changed frame size : 600 x 300```   
   

#### Evaluation on HollywoodHeads Test Dataset   
           
![image (5)](https://user-images.githubusercontent.com/88639269/201271582-48c88999-ae31-4908-b954-8192951bc974.png)


### Using YOLOX

1. Video
```
python facedetect.py video  --name yolox-x  --ckpt ./yolox_x.pth --test_path ./video/HOLLYWOOD.mp4  --save_path ./test_dataset --save y
```

2. Image
```
python facedetect.py image  --name yolox-x --ckpt ./yolox_x.pth --test_path ./test_dataset/test_image/HOLLYWOOD --save_path ./test_dataset --save y 
```


### Using YOLOX + Deepsort

1. Video
```
python facedetect.py video --name deepyolox-x --ckpt ./yolox_x.pth --test_path ./video/HOLLYWOOD.mp4  --tsize 400  --save_path ./test_dataset --save y
```
- Tracking cannot detect the face in a single image. 

### Folder Structure

![image](https://user-images.githubusercontent.com/88639269/201103112-e91dabf0-e2c8-43f8-8203-715b8fece94f.png)


