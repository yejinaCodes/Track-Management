#values each object must have

class Object():
    def __init__(self, init_id=0, keypoints, features, centerpoint, class_type):
        self.id = init_id
        self.keypoints = keypoints
        self.features = features
        self.centerpoint = centerpoint
        self.class_type = class_type
        


    def update_id(self):
        return self.id + 1