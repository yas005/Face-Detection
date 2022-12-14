# Face Detection and Tracking (YOLOX, YOLOX + DeepSORT)

![ezgif com-gif-maker](https://user-images.githubusercontent.com/44921488/192144865-cf819b6b-c066-417f-8662-35c59fce4677.gif)

## 1. Get Ready
dataset download : [HollywoodHeads Dataset](https://drive.google.com/file/d/1T5LWSezp2xeSqr1GBskOtBczXQ8ew6q7/view?usp=sharing)  
Wider-face trained YOLOX-X Weight download : [yolox_x.pth](https://drive.google.com/file/d/17U4TgZf7crBV8yZ1kGHt0MeU6hLn2Vfl/view?usp=sharing)    

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
#### YOLOX-X mAP : 70.35% (Train: Wider Face, Test: HollywoodHeads)

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


#### Result:
If the image size is resized smaller than the original image, performance(mAP matrix value) will be lower as expected. 
However, the performance does not deteriorate much.    
If the image size is resized larger than the original image, it is similar to the original image performance.(be almost same) 

[ mov_001_030700.jpeg ]

![mov_001_030700_224](https://user-images.githubusercontent.com/88639269/197601789-95582461-4ccc-4507-92e4-6302e18d35ca.jpeg)
![mov_001_030700](https://user-images.githubusercontent.com/88639269/197601733-0b79ae36-7f9a-45a0-a481-10781ee946ca.jpeg)

#### [options]

```
1. video / image : input??? ????????????, ????????? ??????????????? ??????
2. --tsize : frame(?????????) ?????? ?????? (int)
3. --test_path : video ?????? ?????? ??????/ image ?????? ??????
4. --name : ?????? ?????? 
5. --ckpt : widerface??? ????????? yolox??? ????????? ??????
6. --save_path : face detection ?????????, ?????? ?????? ??????
7. --save : ????????????????????? ?????? ??????(y/n)
```
#### How to use the options (base)

python facedetect.py  ```(video/image)```  --name ```"????????? ??????-backbone??????"``` --ckpt ```"????????? ?????? ?????????"```  --test_path   ```"????????????/????????? ??????"``` 

--save_path ```"???????????? ?????????/????????? ????????? ??????"``` --save ```"???????????? ????????? ?????? ??????(y/n)"```


#### --tsize (resize)

- yolox model : cannot resize image ```( cannot use --tsize )``` : fixed at 640  
- yolox+deepsort model : can resize image ```( can use --tsize )```   
- code : ```im = imutils.resize(im, height=self.resize)```    
Q. Why do you only get a height value?  
A : When the height value is received, the size is changed based on the width and height proportional values of the original frame.     
ex) ```original frame size : 400 x 200```     &rarr;     ```--tsize 300```     &rarr;    ```changed frame size : 600 x 300```   
