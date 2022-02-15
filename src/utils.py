import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def load_video(path, show=False, f=1):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    frames = []
    while(ret):
        frame = cv2.resize(frame, (0,0), fx=f, fy=f)
        frames.append(frame)
        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                show = False
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return frames

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def get_points(img):
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_output(path='./output/', framerate=30, output_name='output.mp4'):
    return os.system(f"ffmpeg -r {framerate} -i {path}%4d.jpg -crf 8 -c:v libx264 -y {output_name}")

def clean_frames(frames, background, k=5):
    """
    @brief  Apply morphological operations to remove noise and background.
    @param  frames      list of frames
    @param  background  averaged background
    @param  k           kernel size (default 5)
    @return             list of cleaned frames
    @return             list of contours sorted by area per frame
    """
    approx_objects = []
    sorted_contours = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k),int(k)))
    for frame in frames:
        bg_remove = cv2.absdiff(
            cv2.cvtColor(cv2.medianBlur(frame,7), cv2.COLOR_BGR2GRAY),
            cv2.medianBlur(background,7)
            )
        _, approx_obj = cv2.threshold(bg_remove, 0, 255, cv2.THRESH_OTSU)
        approx_obj = cv2.morphologyEx(approx_obj, cv2.MORPH_OPEN, kernel)
        approx_objects.append(approx_obj)

        contours, hierarchy = cv2.findContours(approx_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_area = np.array([cv2.contourArea(c) for c in contours])
        cont_areas = []
        for c in contours:
            cont_areas.append(cv2.contourArea(c))
        scontours = sorted(zip(cont_areas, contours), key=lambda x: x[0], reverse=True)
        sorted_contours.append(scontours)
    return approx_objects, sorted_contours

def get_background(frames):
    """
    @brief  Extract the stationary background of a video.
    @param  frames: list of frames
    @return averaged background
    """
    indices = np.random.randint(0, len(frames), size=30)
    sample_frames = [frames[i] for i in indices]
    avg_frames = np.median(sample_frames, axis=0).astype(np.uint8)
    background = cv2.cvtColor(avg_frames, cv2.COLOR_BGR2GRAY)
    return background

def get_index_cmap(count, cmap_name='viridis'):
    """
    @brief  Get the colours of indexed contours.
    @param  count:      number of contours
    @param  cmap_name:  colormap (default 'viridis')
    @return colour of contour
    """
    colours = []
    cmap = plt.cm.get_cmap(cmap_name)
    if count == 1:
        v = count
        r, g, b = [int(x) for x in cmap(v, bytes=True)[:3]]
        colours.append((b, g, r))
        return colours
    for i in range(count):
        v = i / (count - 1)
        r, g, b = [int(x) for x in cmap(v, bytes=True)[:3]]
        colours.append((b, g, r))
    return colours



