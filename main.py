from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssign
from cam_movement_est import Camera
from view_transforms import ViewTransformer
from speed_dist_calc import Speed_dist_estimator
import cv2
import numpy as np

def main():
    vid_frames = read_video(r'D:\projects\2. Computer Vision\yolo-football\input_vids\vid3.mp4')

    #init tracker
    tracker = Tracker(r'D:\projects\2. Computer Vision\yolo-football\models\best.pt')
    tracks = tracker.get_tracks(vid_frames,
                                read_from_stub=True,
                                stub_path='stubs/track_stubs.pkl')

    #get positions
    tracker.add_position_to_tracks(tracks)  #get positions of objects to adjust with camera movement
    
    #save cropped img of player by bbox


    #cam movement est
    camera_movement_est = Camera(vid_frames[0])
    camera_movement_per_frame = camera_movement_est.get_movement(vid_frames,read_from_stub=True,stub_path='stubs/cam_movement.pkl')
    camera_movement_est.add_adjusted_positions_to_tracks(tracks,camera_movement_per_frame)

    #view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_positions_to_tracks(tracks)

    #interpolate posns
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])
    
    #speed dist estimator
    speed_dist_estimator = Speed_dist_estimator()
    speed_dist_estimator.add_speed_and_dist_to_tracks(tracks)

    #assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(vid_frames[0],tracks['players'][0])

    #find colors and team for each player and put into dict for later use
    for frameno,player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(vid_frames[frameno],track['bbox'],player_id)
            tracks['players'][frameno][player_id]['team'] = team            #new attributes for detected players
            tracks['players'][frameno][player_id]['team_color'] = team_assigner.team_colors[team]

    #assign ball acquisition
    player_assigner = PlayerBallAssign()
    team_ball_control = []

    for frameno,player in enumerate(tracks['players']):
        ball_box = tracks['ball'][frameno][1]['bbox']
        assigned_player = player_assigner.assign_ball(player,ball_box)

        if assigned_player !=-1:
            tracks['players'][frameno][assigned_player]['has_ball'] = True #new attribute
            team_ball_control.append(tracks['players'][frameno][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1]) #if ball in air, control goes to last owner

    team_ball_control = np.array(team_ball_control)


    #draw output
    op_vid = tracker.draw_annotaions(vid_frames,tracks,team_ball_control)

    #draw cam movement
    op_vid = camera_movement_est.draw_camera_movement(op_vid,camera_movement_per_frame)

    #draw speed dist
    speed_dist_estimator.draw_speed_dist(op_vid,tracks)

    save_video(op_vid,r'D:\projects\2. Computer Vision\yolo-football\output_vids\test_output.avi')

if __name__=='__main__':
    main()