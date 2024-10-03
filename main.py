import os
import cv2
import glob
import constants
import pandas as pd
from copy import deepcopy
from moviepy.editor import *
from mini_court import MiniCourt
from hawkeye import in_out_checker
from detectors import BounceDetector
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from utils import (read_video, save_video, measure_distance,
                   draw_player_stats, convert_pixel_distance_to_meters)

input_video_folder_path = r"input_videos"
output_video_folder_path = r"output_videos"

def main():
    # Getting all videos from the input directory
    input_videos = glob.glob(f"{input_video_folder_path}/*")
    if len(input_videos) == 0:
        print("No video is present in the input folder")
    else:    
        for video_path in input_videos:
            # Read Video
            if not os.path.exists('tmp'):
                os.mkdir('tmp')
            
            input_video_path = f"{video_path}"
            clip = VideoFileClip(f"{video_path}")
            fps = int(clip.fps)
            video_frames = read_video(input_video_path, fps)
            # Extracting audio
            audioclip = AudioFileClip(input_video_path)
            new_audioclip = CompositeAudioClip([audioclip])
            
            # Detect Players, Ball and Bounce
            player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
            ball_tracker = BallTracker(model_path='models/tracknet_model_best.pt') 
            bounce_detector = BounceDetector(model_path='models/ctb_regr_bounce.cbm')
        
            player_detections = player_tracker.detect_frames(video_frames,
                                                               read_from_stub=False,
                                           stub_path="tracker_stubs/player_detections.pkl"
                                        )
            ball_detections = ball_tracker.detect_frames(video_frames, 
                                                         read_from_stub=False,
                                        stub_path="tracker_stubs/ball_detections.pkl"
                                        )
            ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
            bounce_detection_frames = bounce_detector.detect_bounce_positions(ball_detections)

            # Court Line Detector model
            court_model_path = "models/keypoints_model_trained.pth"
            court_line_detector = CourtLineDetector(court_model_path)
            court_keypoints = court_line_detector.predict(video_frames[0])
        
            # choose players
            player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        
            # MiniCourt
            mini_court = MiniCourt(video_frames[0]) 
        
            # Detect ball shots
            ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
            # Convert positions to mini court positions
            player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                                  ball_detections,
                                                                                                                  court_keypoints)
            
            player_depth_data = [{
                'frame_num':0,
                'player_1_last_shot_depth':0,
                'player_1_total_shot_depth':0,
                'player_1_total_bounce':0.000001,
                'player_1_hawkeye':0,
                'player_2_last_shot_depth':0,
                'player_2_total_shot_depth':0,
                'player_2_total_bounce':0.000001,
                'player_2_hawkeye':0
            } ]

            for bounce_shot_ind in range(len(bounce_detection_frames)):
                # Get distance covered by the ball
                bounce_frame = bounce_detection_frames[bounce_shot_ind]
                ball_player1_distance = measure_distance(player_mini_court_detections[bounce_frame][1],
                                                                ball_mini_court_detections[bounce_frame][1])
                
                ball_player2_distance = measure_distance(player_mini_court_detections[bounce_frame][2],
                                                                ball_mini_court_detections[bounce_frame][1])
                
                if ball_player1_distance > ball_player2_distance:
                    player_depth_shot_id = 1
                else:
                    player_depth_shot_id = 2
                
                tennis_net_y = mini_court.get_tennis_net_y_point()
                depth_of_shot = abs(tennis_net_y - ball_mini_court_detections[bounce_frame][1][1])

                depth_meters = convert_pixel_distance_to_meters( depth_of_shot,
                                                                constants.DOUBLE_LINE_WIDTH,
                                                                mini_court.get_width_of_mini_court()
                                                                )
                # For Hawk eye 0:IN, 1:OUT
                mini_court_keypoints_hwk = mini_court.get_court_drawing_keypoints()
                hawk_eye = in_out_checker(mini_court_keypoints_hwk, ball_mini_court_detections[bounce_frame][1])

                current_player_depth = deepcopy(player_depth_data[-1])
                current_player_depth['frame_num'] = bounce_frame
                current_player_depth[f'player_{player_depth_shot_id}_total_bounce'] += 1
                current_player_depth[f'player_{player_depth_shot_id}_last_shot_depth'] = depth_meters
                current_player_depth[f'player_{player_depth_shot_id}_total_shot_depth'] += depth_meters
                current_player_depth[f'player_{player_depth_shot_id}_hawkeye'] = hawk_eye
                player_depth_data.append(current_player_depth)
            player_depth_data_df = pd.DataFrame(player_depth_data)               


            player_stats_data = [{
                'frame_num':0,
                'player_1_number_of_shots':0.000001,
                'player_1_total_shot_speed':0,
                'player_1_last_shot_speed':0,
                'player_1_total_player_speed':0,
                'player_1_last_player_speed':0,
        
                'player_2_number_of_shots':0.000001,
                'player_2_total_shot_speed':0,
                'player_2_last_shot_speed':0,
                'player_2_total_player_speed':0,
                'player_2_last_player_speed':0,
            } ]
            
            for ball_shot_ind in range(len(ball_shot_frames)-1):
                try:
                    start_frame = ball_shot_frames[ball_shot_ind]
                    end_frame = ball_shot_frames[ball_shot_ind+1]
                    ball_shot_time_in_seconds = (end_frame-start_frame)/fps # 24fps
        
                    # Get distance covered by the ball
                    distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                                    ball_mini_court_detections[end_frame][1])
                    distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                                    constants.DOUBLE_LINE_WIDTH,
                                                                                    mini_court.get_width_of_mini_court()
                                                                                    ) 
        
                    # Speed of the ball shot in km/h
                    speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6
        
                    # player who the ball
                    player_positions = player_mini_court_detections[start_frame]
                    player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                            ball_mini_court_detections[start_frame][1]))
        
                    # opponent player speed
                    opponent_player_id = 1 if player_shot_ball == 2 else 2
                    distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                            player_mini_court_detections[end_frame][opponent_player_id])
                    distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                                    constants.DOUBLE_LINE_WIDTH,
                                                                                    mini_court.get_width_of_mini_court()
                                                                                    ) 
        
                    speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6
        
                    current_player_stats= deepcopy(player_stats_data[-1])
                    current_player_stats['frame_num'] = start_frame
                    current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
                    current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
                    current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        
                    current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
                    current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent
        
                    player_stats_data.append(current_player_stats)
                except:
                    continue
            
            player_stats_data_df = pd.DataFrame(player_stats_data)
            frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
            player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
            player_stats_data_df = pd.merge(player_stats_data_df, player_depth_data_df, on='frame_num', how='left' )
            player_stats_data_df = player_stats_data_df.ffill()

            player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
            player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
            player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
            player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']
            
            player_stats_data_df['player_1_average_shot_depth'] = player_stats_data_df['player_1_total_shot_depth']/player_stats_data_df['player_1_total_bounce']
            player_stats_data_df['player_2_average_shot_depth'] = player_stats_data_df['player_2_total_shot_depth']/player_stats_data_df['player_2_total_bounce']

        
        
            # Draw output
            # Draw Player Bounding Boxes
            output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
            output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)
            
            # Draw court Keypoints
            output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
        
            # # Draw Mini Court
            output_video_frames = mini_court.draw_mini_court(output_video_frames)
            output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
            output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    
        
            # Draw Player Stats
            output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)
        
            ## Draw frame number on top left corner
            for i, frame in enumerate(output_video_frames):
                cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            # save_video
            input_name = os.path.basename(f"{video_path}")
            
            save_video(output_video_frames, f"tmp\output_{input_name}.avi", fps)

            # Printing average stats
            print(f"---------------For video {input_name}---------------")
            print("player_1_average_shot_speed:",player_stats_data_df['player_1_average_shot_speed'].iloc[-1])
            print("player_2_average_shot_speed:",player_stats_data_df['player_2_average_shot_speed'].iloc[-1])
            print("player_1_average_player_speed:",player_stats_data_df['player_1_average_player_speed'].iloc[-1])
            print("player_2_average_player_speed:",player_stats_data_df['player_2_average_player_speed'].iloc[-1])
            print("player_1_average_shot_depth:",player_stats_data_df['player_1_average_shot_depth'].iloc[-1])
            print("player_2_average_shot_depth:",player_stats_data_df['player_2_average_shot_depth'].iloc[-1])
            
            # avi to mp4
            video_clip_final = VideoFileClip(f"tmp\output_{input_name}.avi")
            path, file_name = os.path.split(f"{output_video_folder_path}/output_{input_name}.avi")
            output_name = os.path.join(path, os.path.splitext(file_name)[0])

            # adding audio
            video_clip_final.audio = new_audioclip
            video_clip_final.write_videofile(output_name, logger=None)
            
            # Deleting redundant files
            files = glob.glob('tmp/*')
            for f in files:
                os.remove(f)
            if os.path.exists('tmp'):
                os.rmdir('tmp')    
            

if __name__ == "__main__":
    main()