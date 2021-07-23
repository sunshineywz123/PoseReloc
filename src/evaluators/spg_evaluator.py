import numpy as np

class Evaluator():
    def __init__(self):
        self.cmd5 = []
        self.add = []
    
    def add_metric(self, pose_pred, pose_target, percentage=0.1):
        pass

    def cm_degree_5_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)
    
    def evaluate(self, pose_pred, pose_gt):
        if pose_pred is None:
            self.cmd5.append(False)
        else:
            if pose_pred.shape == (4, 4):
                pose_pred = pose_pred[:3, :4]
            if pose_gt.shape == (4, 4):
                pose_gt = pose_gt[:3, :4]
            self.cm_degree_5_metric(pose_pred, pose_gt)
    
    def summarize(self):
        cmd5 = np.mean(self.cmd5)
        print('5 cm 5 degree metric: {}'.format(cmd5))

        self.cmd5 = []
        return {'cmd5': cmd5}