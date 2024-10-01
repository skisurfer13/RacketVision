import cv2
import os
import moviepy.editor as mp

def read_video(video_path, fps):
    clip = mp.VideoFileClip(video_path)
    dim = clip.size

    factor = 1920/dim[0]
    clip_resized = clip.resize(factor)
    path, vid_name = os.path.split(video_path)
    clip_resized.write_videofile(f"tmp\{vid_name}", fps)
    cap = cv2.VideoCapture(f"tmp\{vid_name}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()