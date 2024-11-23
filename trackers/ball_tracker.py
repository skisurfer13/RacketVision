import cv2
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import distance
from model import BallTrackerNet

class BallTracker:
    def __init__(self,model_path):
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 15
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[df_ball_positions.index[i], 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        ball_track, dists = self.detect_frame(frames)
        ball_detections = self.remove_outliers(ball_track, dists)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
                
        return ball_detections

    def postprocess(self, feature_map, original_width):
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                maxRadius=7)
        x,y = None, None
        scale = original_width//640
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
    
    def remove_outliers(self, ball_track, dists, max_dist = 200):
        # Remove outliers from model prediction    
        # :params
        #     ball_track: list of detected ball points
        #     dists: list of euclidean distances between two neighbouring ball points
        #     max_dist: maximum distance between two neighbouring ball points
        # :return
        #     ball_track: list of ball points
        dist_array = np.array(dists)

        np_where = np.where(dist_array > max_dist)
        outliers = list(np_where[0])

        for i in outliers:
            if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i-1] == -1:
                ball_track[i-1] = (None, None)
        
        ball_detections = []
        for ball in ball_track:
            ball_detections.append({1:[ball[0], ball[1], ball[0], ball[1]]})
        return ball_detections

    def detect_frame(self,frames):
        dists = [-1]*2
        ball_track = [(None,None)]*2
        height = 360
        width = 640
        origina_height, original_width = frames[0].shape[:2]
        for num in tqdm(range(2, len(frames))):

            img = cv2.resize(frames[num], (width, height))
            img_prev = cv2.resize(frames[num-1], (width, height))
            img_preprev = cv2.resize(frames[num-2], (width, height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            out = self.model(torch.from_numpy(inp).float().to('cpu'))
            out = out.argmax(dim=1)
            output = out.detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, original_width)
            
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)  

        return ball_track, dists 

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                if bbox[0] is not None and x1 is not None:
                    cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames


    