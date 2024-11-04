import sys
sys.path.append('../')
from utils import get_box_centre,measure_dist

class PlayerBallAssign():
    def __init__(self):
        self.max_dist = 70

    def assign_ball(self,players,ball_box):
        ball_pos = get_box_centre(ball_box)

        mini = 9999999999999
        assigned_player = -1

        for p_id,player in players.items():
            player_box = player['bbox']

            #take bottom 2 corners, assume legs are there
            dist_left = measure_dist((player_box[0],player_box[3]),ball_pos)
            dist_right = measure_dist((player_box[2],player_box[3]),ball_pos)
            dist = min(dist_left,dist_right)

            if dist<self.max_dist:
                if dist<mini:
                    mini = dist
                    assigned_player = p_id

        return assigned_player