import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import ce_module as ce 
import os 
import glob

behav_cam_directory = sys.argv[1]

file_ends_with = sys.argv[2]

file_list = [behav_cam_directory+f for f in os.listdir(behav_cam_directory) if f.endswith(file_ends_with)]


for path_to_movie in file_list:
    print('loading and converting movie to grayscale')
    grayscale_frames = ce.load_and_convert_movie(path_to_movie)
    plt.imshow(grayscale_frames[2], cmap='gray')
    plt.show()

    print('saving results')
    normalized_results = ce.save_grayscale_movie(grayscale_frames, path_to_movie)


    print('enhancing contrast')
    ce_frames_hist  = ce.auto_contrast_adjustment_eqhist(grayscale_frames)

    print('saving results')
    normalized_frames = ce.save_contrast_enhanced_movie(ce_frames_hist, path_to_movie)