from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import pandas as pd
import numpy as np
sys.path.append('../')  #sets path for this specific file
from utils import get_box_centre,get_box_width,get_foot

class Tracker:
    def __init__(self,model):
        self.model = YOLO(model)
        #auto called when creating instance of class
        self.tracker = sv.ByteTrack()
        pass
    
    def add_position_to_tracks(self,tracks):
        for obj,obj_tracks in tracks.items():
            for frameno,track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    box = track_info['bbox']
                    if obj=='ball':
                        position = get_box_centre(box)
                    else:
                        position = get_foot(box)
                    tracks[obj][frameno][track_id]['position'] = position #new attribute
                    
    def interpolate_ball(self,ball_posns):
        #TODO: check this out
        ball_posns = [x.get(1,{}).get('bbox',[]) for x in ball_posns] #idek
        df_posns = pd.DataFrame(ball_posns,columns=['x1','y1','x2','y2'])

        df_posns = df_posns.interpolate() #magik
        df_posns = df_posns.bfill() 

        ball_posns = [{1:{'bbox':x}} for x in df_posns.to_numpy().tolist()] #return to original format
        return ball_posns

    def detect_frames(self,frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+=detections_batch
        return detections
    
    def get_tracks(self, frames, read_from_stub =False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.detect_frames(frames)

        #output
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frameno,frame_detection in enumerate(detections):
            detected_classes = frame_detection.names
            detected_classes_inv = {v:k for k,v in detected_classes.items()}
            #invert the class:obj maps to obj:class for convnience
    
            detection_supervision = sv.Detections.from_ultralytics(frame_detection)

            #assume goalkeeper as player
            for ind,classid in enumerate(detection_supervision.class_id):
                if detected_classes[classid]=='goalkeeper':
                    detection_supervision.class_id[ind] = detected_classes_inv['player']


            #actual tracking
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id==detected_classes_inv['player']:
                    tracks['players'][frameno][track_id] = {"bbox":bbox}
                
                if cls_id==detected_classes_inv['referee']:
                    tracks['referees'][frameno][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id==detected_classes_inv['ball']:
                    tracks['ball'][frameno][1] = {"bbox":bbox}
            
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self,frame,box,color,t_id=None):
        y2 = int(box[3]) #bottom right/left y coord
        x_centre,_ = get_box_centre(box)
        #take centre of x to make box in middle, take any of bottom y
        width = get_box_width(box)

        cv2.ellipse(
            frame,
            center=(x_centre,y2),
            axes=(int(width),int(0.35*width)),  #radii of ellipse
            angle=0,
            startAngle=-45,      #start and end of circumference
            endAngle=235,   
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_centre - rectangle_width//2)
        x2_rect = int(x_centre + rectangle_width//2)
        y1_rect = int((y2-rectangle_height//2) + 15)
        y2_rect = int((y2+rectangle_height//2) + 15)
        #figure this out yourself

        if t_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect,y1_rect), #top left
                (x2_rect,y2_rect), #bottom right
                color,
                cv2.FILLED
                )
            
            x1_text = x1_rect + 12
            if t_id > 9:
                x1_text -=5
            if t_id > 99: #wider number, need to positon seprately
                x1_text -=5
            
            
            cv2.putText(
                frame,
                f"{t_id}",
                (int(x1_text),int(y1_rect+15)), #y is inverted. take top y and add val to display text lower
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self,frame,box,color):
        y = int(box[1])
        x,_ = get_box_centre(box)

        triangle_pts = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])

        cv2.drawContours(frame,[triangle_pts],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_pts],0,(0,0,0), 2)

        return frame

    def draw_control_stats(self,frame,frameno,team_ball_control):
        #semi transparent rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850),(1900,970),(255,255,255),-1)
        alpha = 0.4 #opacity
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        both_team_ball_control_till_frame = team_ball_control[:frameno+1]
        # how many frames each team has control
        team_1_num_frames = both_team_ball_control_till_frame[both_team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = both_team_ball_control_till_frame[both_team_ball_control_till_frame==2].shape[0]

        team1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame,f"Team 1 Ball Control: {team1*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball Control: {team2*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame
    
    def draw_annotaions(self, frames, tracks, team_ball_control):
        op = []

        for frameno, frame in enumerate(frames):
            frame = frame.copy() #do not draw on original
            player_dict = tracks['players'][frameno]
            referee_dict = tracks['referees'][frameno]
            ball_dict = tracks['ball'][frameno]

            #draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color',(0,0,255))  #from modified tracks dict
                frame = self.draw_ellipse(frame,player['bbox'],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))
            
            #draw refs
            for track_id, ref in referee_dict.items():
                frame = self.draw_ellipse(frame,ref['bbox'],(0,255,255))
            
            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball['bbox'],(0,255,0))

            #draw control stats
            frame = self.draw_control_stats(frame,frameno,team_ball_control)

            op.append(frame)
        
        return op
