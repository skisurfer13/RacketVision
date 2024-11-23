
def in_out_checker(mini_court_keypoints, ball_mini_court_coord):
    x_ball_mini = ball_mini_court_coord[0]
    y_ball_mini = ball_mini_court_coord[1]
    
    if (x_ball_mini < mini_court_keypoints[8]) or (x_ball_mini > mini_court_keypoints[12]):
        return 1
    elif (y_ball_mini < mini_court_keypoints[9]) or (y_ball_mini > mini_court_keypoints[11]):
        return 1
    else:
        return 0