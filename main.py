#by concyclics
# -*- coding:UTF-8 -*-

import os
import sys
sys.path.append('./codes/')
import FaceDetect
if __name__=='__main__':
    
    os.chdir('./codes/')
    FaceDetect.faceDetect()#inputPath:str outputPath:str