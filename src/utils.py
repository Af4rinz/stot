import cv2
import os

def load_video(path, show=False, f=0.5):
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
        # init_points.append((x, y))

def get_points(img):
    # global init_points 
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_output(path='./output/', framerate=30, output_name='output.mp4'):
    return os.system(f"ffmpeg -r {framerate} -i {path}%4d.jpg -crf 15 -b 800k -vcodec mpeg4 -y {output_name}")
