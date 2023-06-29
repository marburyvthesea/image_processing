import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing


def process_frame(frame, shared_frames):
    # Convert the frame to grayscale if it's not already
    if len(frame.shape) == 3:  # Color image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    shared_frames.append(frame)

def load_and_convert_movie_mp(filename):
    cap = cv2.VideoCapture(filename)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    manager = multiprocessing.Manager()
    frames = manager.list()  # Use Manager.list() for shared list among processes
    
    with tqdm(total=total_frames, desc='Loading Movie', unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process the frame in a separate process
            process = multiprocessing.Process(target=process_frame, args=(frame, frames))
            process.start()
            process.join()
            
            pbar.update(1)
    
    cap.release()
    return list(frames)  # Convert Manager.list() to a regular list

def load_and_convert_movie(filename):
    cap = cv2.VideoCapture(filename)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    with tqdm(total=total_frames, desc='Loading Movie', unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert the frame to grayscale if it's not already
            if len(frame.shape) == 3:  # Color image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames.append(frame)
            pbar.update(1)
    
    cap.release()
    return frames

def auto_contrast_adjustment(frames, saturation_threshold=0.35):
    adjusted_frames = []
    
    with tqdm(total=len(frames), desc='Processing Frames', unit='frame') as pbar:
        for frame in frames:
            # Perform auto contrast adjustment using the specified saturation threshold
            min_val = np.percentile(frame, saturation_threshold * 100)
            max_val = np.percentile(frame, (1 - saturation_threshold) * 100)
            adjusted_frame = np.clip(frame, min_val, max_val)
            
            # Scale the pixel values to the full dynamic range (0-255)
            adjusted_frame = ((adjusted_frame - min_val) / (max_val - min_val)) * 255
            
            # Convert the pixel values to uint8 data type
            adjusted_frame = adjusted_frame.astype(np.uint8)
            
            adjusted_frames.append(adjusted_frame)
            pbar.update(1)
    
    return adjusted_frames


def auto_contrast_adjustment_eqhist(frames):
    adjusted_frames = []
    
    with tqdm(total=len(frames), desc='Processing Frames', unit='frame') as pbar:
        for frame in frames:
            # Apply histogram equalization to enhance contrast and brightness
            adjusted_frame = cv2.equalizeHist(frame)
            
            adjusted_frames.append(adjusted_frame)
            pbar.update(1)
            
    return adjusted_frames

def process_frame_mp(frame, saturation_threshold):
    # Perform auto contrast adjustment using the specified saturation threshold
    min_val = np.percentile(frame, saturation_threshold * 100)
    max_val = np.percentile(frame, (1 - saturation_threshold) * 100)
    adjusted_frame = np.clip(frame, min_val, max_val)

    # Scale the pixel values to the full dynamic range (0-255)
    adjusted_frame = ((adjusted_frame - min_val) / (max_val - min_val)) * 255

    # Convert the pixel values to uint8 data type
    adjusted_frame = adjusted_frame.astype(np.uint8)

    return adjusted_frame


def auto_contrast_adjustment_mp(frames, saturation_threshold=0.35, process_func=process_frame_mp):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    
    results = []
    with tqdm(total=len(frames), desc='Processing Frames', unit='frame') as pbar:
        for frame in frames:
            result = pool.apply_async(process_func, args=(frame, saturation_threshold))
            results.append(result)
        
        adjusted_frames = [result.get() for result in results]
        pbar.update(len(frames))
    
    pool.close()
    pool.join()
    
    return adjusted_frames


def save_contrast_enhanced_movie(frames, original_filename):
    # Modify the original filename to include "_python_ce" and replace spaces with "_"
    filename_parts = original_filename.split('.')
    new_filename = filename_parts[0].replace(' ', '_') + '_python_ce_grayscale.mp4'

    # Get the frame rate (fps) of the original movie
    cap = cv2.VideoCapture(original_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get the shape of the original frames
    height, width = frames[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_filename, fourcc, fps, (width, height), isColor=False)

    results = []
    with tqdm(total=len(frames), desc='Saving Movie', unit='frame') as pbar:
        for frame in frames:
            # Normalize pixel values to the range of 0-255
            normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out.write(normalized_frame)
            pbar.update(1)
            results.append(normalized_frame)
            
    out.release()
    print(f"Contrast-enhanced movie (grayscale, XVID compression) saved as: {new_filename}")
    return(results)

def save_contrast_enhanced_movie_non_normalized(frames, original_filename):
    # Modify the original filename to include "_python_ce" and replace spaces with "_"
    filename_parts = original_filename.split('.')
    new_filename = filename_parts[0].replace(' ', '_') + '_python_ce_grayscale_nn.mp4'

    # Get the frame rate (fps) of the original movie
    cap = cv2.VideoCapture(original_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get the shape of the original frames
    height, width = frames[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_filename, fourcc, fps, (width, height), isColor=False)

    results = []
    with tqdm(total=len(frames), desc='Saving Movie', unit='frame') as pbar:
        for frame in frames:
            # Normalize pixel values to the range of 0-255
            #normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out.write(frame)
            pbar.update(1)
            #results.append(normalized_frame)
            
    out.release()
    print(f"Contrast-enhanced movie (grayscale, XVID compression) saved as: {new_filename}")
    return(True)

def save_grayscale_movie(frames, original_filename):
    # Modify the original filename to include "_python_ce" and replace spaces with "_"
    filename_parts = original_filename.split('.')
    new_filename = filename_parts[0].replace(' ', '_') + '_python_grayscale.mp4'

    # Get the frame rate (fps) of the original movie
    cap = cv2.VideoCapture(original_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get the shape of the original frames
    height, width = frames[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_filename, fourcc, fps, (width, height), isColor=False)

    results = []
    with tqdm(total=len(frames), desc='Saving Movie', unit='frame') as pbar:
        for frame in frames:
            # Normalize pixel values to the range of 0-255
            normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out.write(normalized_frame)
            pbar.update(1)
            results.append(normalized_frame)
            
    out.release()
    print(f"grayscale movie (grayscale, XVID compression) saved as: {new_filename}")
    return(results)

def save_grayscale_movie_non_normalized(frames, original_filename):
    # Modify the original filename to include "_python_ce" and replace spaces with "_"
    filename_parts = original_filename.split('.')
    new_filename = filename_parts[0].replace(' ', '_') + '_python_grayscale_nn.mp4'

    # Get the frame rate (fps) of the original movie
    cap = cv2.VideoCapture(original_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get the shape of the original frames
    height, width = frames[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_filename, fourcc, fps, (width, height), isColor=False)

    #results = []
    with tqdm(total=len(frames), desc='Saving Movie', unit='frame') as pbar:
        for frame in frames:
            # Normalize pixel values to the range of 0-255
            #normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out.write(frame)
            pbar.update(1)
            #results.append(normalized_frame)
            
    out.release()
    print(f"grayscale movie (grayscale, XVID compression) saved as: {new_filename}")
    return(True)
    
def display_frames(frames):
    for frame in frames:
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.show()