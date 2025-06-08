import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes in the format [x1, y1, x2, y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        bbox format: [x1, y1, x2, y2]
        """
        # Initialize your Kalman filter here
        # For simplicity, I'll omit detailed Kalman filter code
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0
        self.hit_streak += 1

    def predict(self):
        # Update your Kalman filter prediction here
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def get_state(self):
        return self.bbox

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        Returns:
          list of tracks [[x1,y1,x2,y2,id],...]
        """
        self.frame_count += 1

        # Predict new locations of existing trackers
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks.append(pos)
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks.pop(t)

        trks = np.array(trks)
        if len(dets) == 0:
            unmatched_dets = []
            unmatched_trks = list(range(len(trks)))
            matches = np.empty((0, 2), dtype=int)
        else:
            if len(trks) == 0:
                unmatched_dets = list(range(len(dets)))
                unmatched_trks = []
                matches = np.empty((0, 2), dtype=int)
            else:
                matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # Update matched trackers with assigned detections
        for m in matches:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i-1)
            i -= 1

        # Prepare output
        ret = []
        for trk in self.trackers:
            if (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                ret.append(np.concatenate((bbox, [trk.id])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk[:4])

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))

        if matched_indices.size == 0:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(len(detections)):
            if matched_indices.size == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if matched_indices.size == 0 or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matches)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)







