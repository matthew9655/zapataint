import numpy as np


class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.rank = [1] * n

    def find(self, x):
        if self.root[x] == x:
            return x

        self.root[x] = self.find(self.root[x])

        return self.root[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


class track_stats:
    def __init__(self, track):
        self.track = track
        self.start = track.frame_ids[0]
        self.end = track.frame_ids[-1]
        self.n = len(track.frame_ids)

        self.avg_x = sum([bbox[0] for bbox in track.bboxes_traj]) / self.n
        self.avg_y = sum([bbox[1] for bbox in track.bboxes_traj]) / self.n
        self.avg_l = sum([bbox[2] for bbox in track.bboxes_traj]) / self.n
        self.avg_w = sum([bbox[3] for bbox in track.bboxes_traj]) / self.n
        self.avg_yaw = sum([bbox[4] for bbox in track.bboxes_traj]) / self.n

    def __len__(self):
        return len(self.track.frame_ids)


### Helper functions

def line_func(start_x, start_y, yaw):
    m = np.tan(yaw)
    c = start_y - m * start_x
    return -m, -c


def intersect_point(m1, m2, c1, c2):
    """using cramer's rule to return x, y"""
    return (c2 - c1) / (m1 - m2), ((c1 * m2) - (c2 * m1)) / (m1 - m2)


def get_yaw(x1, y1, x2, y2):
    """
    get yaw based on gradient
    """
    m = (y1 - y2) / (x1 - x2)
    return np.arctan(m)
