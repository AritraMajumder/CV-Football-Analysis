import sys
import cv2

sys.path.append('../')
from utils import measure_dist,get_foot

class Speed_dist_estimator():
    def __init__(self):
        self.frame_window = 5 #calc speed every 5 frames
        self.frame_rate = 24

    
    def add_speed_and_dist_to_tracks(self,tracks):
        total_dist = {}

        for obj,obj_tracks in tracks.items():
            if obj=='ball' or obj=='referee':
                continue

            num_frames = len(obj_tracks)
            for frameno in range(0,num_frames,self.frame_window):
                last_frame = min(frameno+self.frame_window,num_frames-1)

                for track_id,_ in obj_tracks[frameno].items():
                    if track_id not in obj_tracks[last_frame]:
                        continue #idk, doesnt much matter
                    
                    start_pos = obj_tracks[frameno][track_id]['position_transformed']
                    end_pos = obj_tracks[last_frame][track_id]['position_transformed']

                    if start_pos is None or end_pos is None:
                        continue

                    distance_covered = measure_dist(start_pos,end_pos)
                    time_elapsed = (last_frame-frameno)/self.frame_rate
                    speed_ms = distance_covered/time_elapsed
                    speed_kmh = speed_ms*3.6

                    if obj not in total_dist:
                        total_dist[obj] = {}
                    
                    if track_id not in total_dist[obj]:
                        total_dist[obj][track_id] = 0
                    
                    total_dist[obj][track_id] +=distance_covered

                    for frameno_batch in range(frameno,last_frame):
                        if track_id not in tracks[obj][frameno_batch]:
                            continue
                        tracks[obj][frameno_batch][track_id]['speed'] = speed_kmh
            
                        tracks[obj][frameno_batch][track_id]['dist'] = total_dist[obj][track_id]
    
    def draw_speed_dist(self,frames,tracks):
        op_frames = []
        for frameno, frame in enumerate(frames):
            for obj,obj_tracks in tracks.items():
                if obj=='ball' or obj =='referee':
                    continue
                for _, track_info in obj_tracks[frameno].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed',None)
                        dist = track_info.get('dist',None)
                        if speed is None or dist is None:
                            continue

                        box = track_info['bbox']
                        posn = get_foot(box)
                        posn = list(posn)
                        posn[1]+=40

                        posn = tuple(map(int,posn))
                        #cv2.putText(frame,f"{speed:.2f}km/h",posn,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                        #cv2.putText(frame,f"{dist:.2f}km/h",(posn[0],posn[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        
            op_frames.append(frame)
        
        return op_frames
