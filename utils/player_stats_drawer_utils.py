import numpy as np
import cv2

def draw_player_stats(output_video_frames,player_stats):

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_forehand_speed = row['player_1_last_forehand_speed']
        player_2_forehand_speed = row['player_2_last_forehand_speed']
        player_1_backhand_speed = row['player_1_last_backhand_speed']
        player_2_backhand_speed = row['player_2_last_backhand_speed']
        player_1_serve_speed = row['player_1_last_serve_speed']
        player_2_serve_speed = row['player_2_last_serve_speed']

        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']
        player_1_depth = row['player_1_last_shot_depth']
        player_2_depth = row['player_2_last_shot_depth']
        player_1_hawkeye = row['player_1_hawkeye']
        player_2_hawkeye = row['player_2_hawkeye']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_forehand_speed = row['player_1_average_forehand_speed']
        avg_player_2_forehand_speed = row['player_2_average_forehand_speed']
        avg_player_1_backhand_speed = row['player_1_average_backhand_speed']
        avg_player_2_backhand_speed = row['player_2_average_backhand_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']
        avg_player_1_shot_depth= row['player_1_average_shot_depth']
        avg_player_2_shot_depth = row['player_2_average_shot_depth']


        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width=350
        height= 490

        start_x = frame.shape[1]-400
        start_y = frame.shape[0]-500
        end_x = start_x+width
        end_y = start_y+height
         # For Hawkeye
        width_hwk = 80
        height_hwk = 70
        start_x_hwk = frame.shape[1]-400
        start_y_hwk = frame.shape[0]-600
        end_x_hwk = start_x_hwk + width_hwk
        end_y_hwk = start_y_hwk + height_hwk

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        
        if player_1_hawkeye == 1 or player_2_hawkeye == 1: 
            cv2.rectangle(overlay, (start_x_hwk, start_y_hwk), (end_x_hwk, end_y_hwk), (0, 0, 0), -1)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Depth of Shot"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_depth:.1f} m    {player_2_depth:.1f} m"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. Shot Depth"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_depth:.1f} m    {avg_player_2_shot_depth:.1f} m"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Forehand Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_forehand_speed:.1f} km/h    {player_2_forehand_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. F.hand Sp."
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+360), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_forehand_speed:.1f} km/h    {avg_player_2_forehand_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Backhand Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+400), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_backhand_speed:.1f} km/h    {player_2_backhand_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. B.hand Sp."
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_backhand_speed:.1f} km/h    {avg_player_2_backhand_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Serve Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+480), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_serve_speed:.1f} km/h    {player_2_serve_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # Hawkeye Text
        if player_1_hawkeye == 1 or player_2_hawkeye == 1: 
            text = "Hawkeye"
            output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x_hwk+8, start_y_hwk+15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
            text = "OUT"
            output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x_hwk+20, start_y_hwk+50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return output_video_frames
