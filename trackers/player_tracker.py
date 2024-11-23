from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path, posture_model_path):
        self.model = YOLO(model_path)
        self.posture_model = YOLO(posture_model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        try:
            chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        except:
            # If cannot find player in the first frame, attempt for another
            chosen_player = self.choose_players(court_keypoints, player_detections[10])
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {}
            for track_id, bbox in player_dict.items():
                if track_id in chosen_player:
                    if track_id == 1:
                        filtered_player_dict[track_id] = bbox
                    else:
                        filtered_player_dict[2] = bbox
            filtered_player_detections.append(filtered_player_dict)
        
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')

            # Inclusion of mid point of 4,6 keypoints for choosing players
            point1_x = (court_keypoints[8] + court_keypoints[12])/2 
            point1_y = (court_keypoints[9] + court_keypoints[13])/2
            court_keypoint_mid = (point1_x, point1_y)
            distance = measure_distance(player_center, court_keypoint_mid)
            if distance < min_distance:
                min_distance = distance

            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self,frames, court_keypoints, read_from_stub=False, stub_path=None):
        player_detections = []
        player_1_posture = []
        player_2_posture = []
        
        # if read_from_stub and stub_path is not None:
        #     with open(stub_path, 'rb') as f:
        #         player_detections = pickle.load(f)
        #     return player_detections
        
        for frame in frames:
            player_dict, player_1_posture_frame, player_2_posture_frame = self.detect_frame(frame, court_keypoints)
            player_detections.append(player_dict)
            player_1_posture.append(player_1_posture_frame[0]) 
            player_2_posture.append(player_2_posture_frame[0]) 
        
        # if stub_path is not None:
        #     with open(stub_path, 'wb') as f:
        #         pickle.dump(player_detections, f)

        return player_detections, player_1_posture, player_2_posture

    def detect_frame(self, frame, court_keypoints):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        
        detected_player_list = []
        for box in results.boxes:
            class_id = int(box.data[0][-1])
            class_name = self.model.names[int(class_id)]
            detected_player_list.append(class_name)


        player_dict = {}
        player_1_posture = []
        player_2_posture = []
        # Incase the players are not detected in the frame
        player_1_x1 = (court_keypoints[10] + court_keypoints[14])/2 - 25
        player_1_x2 = player_1_x1 + 50
        player_1_y2 = (court_keypoints[11] + court_keypoints[15])/2
        player_1_y1 = player_1_y2 - 100
        
        player_2_x1 = (court_keypoints[8] + court_keypoints[12])/2 - 25
        player_2_x2 = player_2_x1 + 50
        player_2_y2 = (court_keypoints[9] + court_keypoints[13])/2
        player_2_y1 = player_2_y2 - 100

        if 'player_1' not in detected_player_list:
            player_1_posture.append('forehand')
            player_dict[1] = [player_1_x1, player_1_y1, player_1_x2, player_1_y2]

        if 'player_2' not in detected_player_list:
            player_2_posture.append('forehand')
            player_dict[2] = [player_2_x1, player_2_y1, player_2_x2, player_2_y2]
                

        # For boxes, predicted in a single frame
        for box in results.boxes:
            if len(box.xyxy.tolist())>0:
                #track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name == 'player_1':
                    player_dict[1] = result
                else:
                    player_dict[2] = result

                # For posture/shot detection of players
                x1, y1, x2, y2 = result
                cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                prediction = self.posture_model.predict(cropped_image)
                posture_dict = prediction[0].names
                posture_index = prediction[0].probs.top1
                
                if object_cls_name == 'player_1':
                    player_1_posture.append(posture_dict[posture_index])
                else:
                    player_2_posture.append(posture_dict[posture_index])

        return player_dict, player_1_posture, player_2_posture

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    