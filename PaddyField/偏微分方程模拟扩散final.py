#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name:    偏微分方程模拟扩散.py
Author:       Liheng-Chan
Email:        202111081084@mail.bnu.edu.cn
Time:         2023/5/21
"""
 
 
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

def my_laplacian(In):
    Out = -1.0 * In + 0.20*(np.roll(In,1,axis=1)+np.roll(In,-1,axis=1)+np.roll(In,1,axis=0)+np.roll(In,-1,axis=0) ) + \
          0.05*(np.roll(np.roll(In,1,axis=0),1,axis=1)+np.roll(np.roll(In,-1,axis=0),1,axis=1)+np.roll(np.roll(In,1,axis=0),-1,axis=1)+np.roll(np.roll(In,-1,axis=0),-1,axis=1))
    return Out

def load_image(file_path, width):
    image = Image.open(file_path).convert("L")  # Convert to grayscale
    image = image.resize((width, width), Image.ANTIALIAS)
    image_array = np.array(image, dtype=float) / 255.0
    return image_array

if __name__=="__main__":
    f = 0.055       #进料率
    k = 0.062       #去除率
    da = 1.0          #U的扩散率
    db = 0.5        #V的扩散率
    width = 128     #网格大小
    dt = 0.25       #每进行一需要0.25秒
    stoptime = 100.0 #自定义模拟时间

    image_file = "C:/image/image6.png"  # Replace with your image file path

    B = load_image(image_file, width)
    A = np.ones((width, width), dtype=float)
    t = 0

    nframes = 1

    while t < stoptime:
        anew = A + (da*my_laplacian(A) - A*(B*B) + f*(1-A))*dt
        bnew = B + (db*my_laplacian(B) + A*(B*B) - (k+f)*B)*dt

        A = anew
        B = bnew
        t += dt
        nframes += 1

    fig, ax0 = plt.subplots()
    f = ax0.pcolor(B)
    fig.colorbar(f, ax=ax0)

    plt.show()

 