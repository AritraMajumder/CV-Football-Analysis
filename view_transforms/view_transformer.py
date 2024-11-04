import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        #we are scanning speeds of players within a trapezoid in the middle of the screen
        court_width = 68
        court_length = 23.32 #105m court divided into 18 strips. take 4 strips 

        #on screen corners
        self.pixel_vertices = np.array(
            [
             [110,1035],
             [265,275],
             [910,260],
             [1640,925]   
            ]
        )
        #actual corners
        self.target_vertices = np.array(
            [
                [0,court_width],
                [0,0],
                [court_length,0],
                [court_length,court_width]

            ]
        )

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)


        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices,self.target_vertices)
    
    def transform_point(self,point):
        p = (int(point[0]),int(point[0]))
        # TODO: check this function out
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >=0 #counting speed if point inside trapezoid
        if not is_inside:
            return None
        
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return transformed_point.reshape(-1,2)

    def add_transformed_positions_to_tracks(self,tracks):
        #takes positions in tracks which are in screen perspective
        #and converts to actual perspective
        for obj,obj_tracks in tracks.items():
            for frameno,track in enumerate(obj_tracks):
                for track_id,track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[obj][frameno][track_id]['position_transformed'] = position_transformed
