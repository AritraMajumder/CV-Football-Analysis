import pickle
import cv2
import numpy as np
import sys
sys.path.append('../')
import os
from utils import measure_dist,measure_xy

# TODO: document this whole process

class Camera():
    def __init__(self,frame):
        self.min_dist = 5
        self.lk_params =dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )
        first_frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features,
        )
    
    def add_adjusted_positions_to_tracks(self,tracks,cam_movement_per_frame):
        for obj,obj_tracks in tracks.items():
            for frameno, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = cam_movement_per_frame[frameno]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[obj][frameno][track_id]['position_adjusted'] = position_adjusted

    def get_movement(self,frames,read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
            
            
        camera_movement= [[0,0]]*len(frames) 

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frameno in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frameno],cv2.COLOR_RGB2GRAY)
            new_features,_,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_dist = 0

            cam_move_x,cam_move_y = 0,0

            for i,(new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_dist(new_features_point,old_features_point)
                if distance>max_dist:
                    max_dist = distance
                    cam_move_x,cam_move_y = measure_xy(old_features_point,new_features_point)
            
            if max_dist>self.min_dist:
                camera_movement[frameno] = [cam_move_x,cam_move_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)
                  
        return camera_movement
    
    def draw_camera_movement(self,frames,camera_movement_per_frame):
        op_frames = []

        for frameno,frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
            x_movement,y_movement = camera_movement_per_frame[frameno]
            frame = cv2.putText(frame,f"Camera Movement x: {x_movement: .2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement y: {y_movement: .2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            op_frames.append(frame)

        return op_frames