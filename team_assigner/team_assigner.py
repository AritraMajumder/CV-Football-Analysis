from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_model(self,image):
        img2d = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,init = 'k-means++',n_init=1)
        kmeans.fit(img2d)

        return kmeans

    def get_col(self,frame,box):
        img = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        top = img[0:int(img.shape[0]/2),:]

        #cluster model
        kmeans = self.get_model(top)

        #cluster labels
        labels = kmeans.labels_
        clustered_img = labels.reshape(top.shape[0],top.shape[1])
        corners = [clustered_img[0,0],clustered_img[0,-1],clustered_img[-1,0],clustered_img[-1,-1]]
        bg_cluster_label = max(set(corners),key = corners.count)
        player_cluster_label = 1-bg_cluster_label

        #color
        col = kmeans.cluster_centers_[player_cluster_label]
        return col

    def assign_team_color(self,frame,player_detections):
        player_colors = [] #all detected player colors in every frame

        for _,player_detection in player_detections.items():
            #iterate over detected player details of the frame
            box = player_detection['bbox']
            col = self.get_col(frame,box)
            player_colors.append(col) 
            #in each iteration detected color added
            #done for all frames
        
        kmeans = KMeans(n_clusters=2,init = 'k-means++',n_init=10)
        kmeans.fit(player_colors) #separates all colors to 2 classes which are used as teams

        self.kmeans = kmeans  #save color detecting model
        self.team_colors[1] = kmeans.cluster_centers_[0] #assign color to team num
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self,frame,box,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_col(frame,box)
 
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] #give color to model, model returns class
        team_id +=1 

        if player_id==84:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id


