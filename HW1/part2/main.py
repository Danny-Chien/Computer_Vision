from importlib.resources import path
from pickletools import float8
from turtle import shape
import numpy as np
import cv2
import argparse
import os

from torch import FloatTensor
from JBF import Joint_bilateral_filter

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    # parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    # parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # read file
    weight = np.zeros(shape=(5, 3))
    try:
        with open(args.setting_path) as f:
            i = 0
            lines = f.readlines()
            for line in lines[1: -1]:
                val = line.split(',')
                weight[i][0] = float(val[0])
                weight[i][1] = float(val[1])
                weight[i][2] = float(val[2])
                i += 1
            tmp = lines[-1].split(',')    
            parser.add_argument('--sigma_s', default=int(tmp[1]), type=int, help='sigma of spatial kernel')
            parser.add_argument('--sigma_r', default=float(tmp[3]), type=float, help='sigma of range kernel') 
            args = parser.parse_args()    
             
    except IOError:
        print('ERROR: can not found ' + args.setting_path)
    # use differnet weight
    JBF = Joint_bilateral_filter(args.sigma_s, args.sigma_r)
    bf = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    for j in range(6):
        if j != 0:
            img_gray = img_rgb[:, :, 0] * weight[j-1][0] + img_rgb[:, :, 1] * weight[j-1][1] + img_rgb[:, :, 2] * weight[j-1][2]
        
        out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        os.chdir(os.path.curdir)
        cost = np.sum(np.abs(out.astype('int32')-bf.astype('int32')))
        cv2.imwrite(f'./data/gray1_{j}.png', img_gray)
        cv2.imwrite(f'./data/rgb1_{j}.png', out)
        print(cost)

if __name__ == '__main__':
    main()