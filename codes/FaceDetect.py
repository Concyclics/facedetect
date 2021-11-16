#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import pathlib
import logging
import cv2
from tools.train_detect import MtcnnDetector
from tqdm import tqdm, trange

logger = logging.getLogger("app")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler.formatter = formatter  # 也可以直接给formatter赋值


def draw_images(img, bboxs, landmarks):  # 在图片上绘制人脸框及特征点
    num_face = bboxs.shape[0]
    for i in range(num_face):
        cv2.rectangle(img, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(
            bboxs[i, 2]), int(bboxs[i, 3])), (0, 255, 0), 3)
    for p in landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[2 * i]), int(p[2 * i + 1])), 3, (0, 0, 255), -1)
    return img


def faceDetect(inputPath:str='../Put_your_images_here/', outputPath:str='../Your_results/'):
    mtcnn_detector = MtcnnDetector(p_model_path="./results/pnet/model_050.pth",
                                   r_model_path="./results/rnet/model_050.pth",
                                   o_model_path="./results/onet/model_050.pth",
                                   min_face_size=24,
                                   use_cuda=False)   # 加载模型参数，构造检测器
    
    inputPath=pathlib.Path(inputPath)
    outputPath=pathlib.Path(outputPath)
    
    logger.info("Init the MtcnnDetector.")
    outputPath.mkdir(exist_ok=True)

    start = time.time()
    for num, input_img_filename in enumerate(inputPath.iterdir()):
        
        img_name = input_img_filename.name
        
        if img_name.endswith('.jpg') is False and img_name.endswith('.JPG') is False and img_name.endswith('.jpeg') is False and img_name.endswith('.JPEG') is False:
            continue
        
        logger.info("Start to process No.{} image.".format(num))
        logger.info("The name of the image is {}.".format(img_name))

        img = cv2.imread(str(input_img_filename))
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxs, landmarks = mtcnn_detector.detect_face(RGB_image)  # 检测得到bboxs以及特征点
        img = draw_images(img, bboxs, landmarks)  # 得到绘制人脸框及特征点的图片
        savePath = outputPath / img_name  # 图片保存路径
        logger.info("Process complete. Save image to {}.".format(str(savePath)))

        cv2.imwrite(str(savePath), img)  # 保存图片

    logger.info("Finish all the images.")
    logger.info("Elapsed time: {:.3f}s".format(time.time() - start))
    
    
def cutvideo(path:str):
    video=cv2.VideoCapture(path)
    w=int(video.get(3))
    h=int(video.get(4))
    fps=int(video.get(5))
    print(w,h,fps)
    frames=[]
    success, frame=video.read()
    while success:
        frames.append(frame)
        success, frame=video.read()
        
    return w,h,fps,frames


def faceDetectVideo(inputPath:str='../Put_your_videos_here/', outputPath:str='../Your_results/'):
    mtcnn_detector = MtcnnDetector(p_model_path="./results/pnet/model_050.pth",
                                   r_model_path="./results/rnet/model_050.pth",
                                   o_model_path="./results/onet/model_050.pth",
                                   min_face_size=24,
                                   use_cuda=False)   # 加载模型参数，构造检测器
    
    inputPath=pathlib.Path(inputPath)
    outputPath=pathlib.Path(outputPath)
    
    logger.info("Init the MtcnnDetector.")
    outputPath.mkdir(exist_ok=True)
    
    for num, input_video_filename in enumerate(inputPath.iterdir()):
        video_name= input_video_filename.name
        if video_name.endswith('.mp4') is False and video_name.endswith('.MOV') is False:
            continue
        
        logger.info("Start to process No.{} video.".format(num))
        logger.info("The name of video is {}.".format(video_name))
        savePath = outputPath / video_name
        
        video=cv2.VideoCapture(str(input_video_filename))
        Width=int(video.get(3))
        Height=int(video.get(4))
        fps=int(video.get(5))
        Total_Frames=int(video.get(7))
        
        writer = cv2.VideoWriter(str(savePath), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (Width, Height), True)
        for cnt in trange(Total_Frames):
            success, img=video.read()
            RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxs, landmarks = mtcnn_detector.detect_face(RGB_image)  # 检测得到bboxs以及特征点
            img = draw_images(img, bboxs, landmarks)  # 得到绘制人脸框及特征点的图片
            writer.write(img)
        
        writer.release()
        
        logger.info("Process complete. Save video to {}.".format(str(savePath)))
        
        
        
        
if __name__=='__main__':
    faceDetectVideo()