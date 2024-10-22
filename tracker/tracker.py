from ultralytics import YOLO

class Tracker:
    def __init__(self, weights="yolov8s-pose.pt", tracker="bytetrack.yaml", device="cpu"):
        self.model = YOLO(weights)
        self.tracker = tracker
        self.device = device

    def get_pose_track(self, frame, conf=0.5):
        result = self.model.track(frame, conf = conf, save = True, tracker = self.tracker, persist=True, device = self.device)[0]

        return result