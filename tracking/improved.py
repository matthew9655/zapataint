from collections import defaultdict
from operator import itemgetter

import numpy as np
import torch

from tracking.improved_types import (
    UnionFind,
    get_yaw,
    intersect_point,
    line_func,
    track_stats,
)

class occlusion_handler:
    def __init__(
        self, tracks, rad_thres=70 * np.pi / 180, score_thres=1, dist_score_divider=5
    ):
        self.total_frames = 80
        self.score_thres = score_thres
        self.rad_thres = rad_thres
        self.tracks = tracks
        self.dist_score_divider = dist_score_divider

        self.target_stats = {}
        self.chosen_stats = {}

    def score(self, track1, track2):
        """
        cost function C1
        """
        # calculate euclidean distance
        length_score = np.sqrt(
            (track1.avg_l - track2.avg_l) ** 2 + (track1.avg_w - track2.avg_w) ** 2
        )
        # calculate frame dist cost
        if track1.start < track2.start:
            frame_dist_score = track2.start - track1.start
        else:
            frame_dist_score = track1.start - track2.end
        frame_dist_score /= self.dist_score_divider

        return length_score + frame_dist_score

    def score1(self, track1, track2):
        """
        cost function C2
        """
        # calculate euclidean distance
        length_score = np.sqrt(
            (track1.avg_l - track2.avg_l) ** 2 + (track1.avg_w - track2.avg_w) ** 2
        )
        position_score = (
            np.sqrt(
                (track1.avg_x - track2.avg_x) ** 2 + (track1.avg_y - track2.avg_y) ** 2
            )
            / 10
        )
        yaw_score = np.sqrt((track1.avg_yaw - track2.avg_yaw) ** 2)

        # calculate frame dist score
        if track1.start < track2.start:
            frame_dist_score = track2.start - track1.start
        else:
            frame_dist_score = track1.start - track2.end
        frame_dist_score /= self.dist_score_divider

        return length_score + position_score + yaw_score + frame_dist_score

    def interp(self, target_bbox, chosen_bbox, track_diff):
        """
        linear interpolation
        """
        target_bbox = target_bbox.numpy()
        chosen_bbox = chosen_bbox.numpy()
        interp_x = np.linspace(target_bbox[0], chosen_bbox[0], track_diff + 1)[1:-1]

        # yaw difference is less than self.rad_thres, normal interpolation
        if abs(target_bbox[4] - chosen_bbox[4]) < self.rad_thres:
            if target_bbox[0] <= chosen_bbox[0]:
                xp = np.asarray([target_bbox[0], chosen_bbox[0]])
                fp = np.asarray([target_bbox[1], chosen_bbox[1]])
            else:
                xp = np.asarray([chosen_bbox[0], target_bbox[0]])
                fp = np.asarray([chosen_bbox[1], target_bbox[1]])

            interp_y = np.interp(interp_x, xp, fp)
            interp_yaw = get_yaw(interp_x, interp_y, chosen_bbox[0], chosen_bbox[1])
        else:
            # yaw difference is greater than self.rad_thres, use intersection as midpoint
            # for interpolation to replicate a turn

            # find intersection of the line between those 2 points
            m1, c1 = line_func(target_bbox[0], target_bbox[1], target_bbox[4])
            m2, c2 = line_func(chosen_bbox[0], chosen_bbox[1], chosen_bbox[4])
            int_x, int_y = intersect_point(m1, m2, c1, c2)
            int_yaw = (
                get_yaw(target_bbox[0], target_bbox[1], int_x, int_y)
                + get_yaw(chosen_bbox[0], chosen_bbox[1], int_x, int_y)
            ) / 2

            # interp with the intersection being the middle
            mid = interp_x.shape[0] // 2
            interp_x[mid] = int_x

            if target_bbox[0] <= chosen_bbox[0]:
                xp = np.asarray([target_bbox[0], int_x, chosen_bbox[0]])
                fp = np.asarray([target_bbox[1], int_y, chosen_bbox[1]])
            else:
                xp = np.asarray([chosen_bbox[0], int_x, target_bbox[0]])
                fp = np.asarray([chosen_bbox[1], int_y, target_bbox[1]])

            interp_y = np.interp(interp_x, xp, fp)

            # calculate yaws
            interp_yaw = np.zeros(interp_x.shape[0])
            for i in range(interp_x.shape[0]):
                if i < mid:
                    interp_yaw[i] = get_yaw(interp_x[i], interp_y[i], int_x, int_y)
                if i > mid:
                    interp_yaw[i] = get_yaw(
                        interp_x[i], interp_y[i], chosen_bbox[0], chosen_bbox[1]
                    )
                else:
                    interp_yaw[i] = int_yaw

        # create list of new bboxes_trajs
        new_bboxes = []
        for i in range(interp_x.shape[0]):
            temp_tensor = torch.tensor(
                [
                    interp_x[i],
                    interp_y[i],
                    chosen_bbox[2],
                    chosen_bbox[3],
                    interp_yaw[i],
                ]
            )
            new_bboxes.append(temp_tensor)

        return new_bboxes

    def fill(self):
        """
        fill in gapped tracklets with linear interpolation
        """
        keys = self.tracks.keys()

        for key in keys:
            # sort bboxes_traj and scores based on frame_ids
            track = self.tracks[key]
            sorted_bboxes_trajs = sorted(
                zip(track.frame_ids, track.bboxes_traj), key=itemgetter(0)
            )
            sorted_scores = sorted(
                zip(track.frame_ids, track.scores), key=itemgetter(0)
            )
            sframe_num = [t[0] for t in sorted_bboxes_trajs]
            sbboxes_traj = [t[1] for t in sorted_bboxes_trajs]
            sscores = [t[1] for t in sorted_scores]

            _, sscores = (
                list(t) for t in zip(*sorted(zip(track.frame_ids, track.scores)))
            )

            for i in range(1, len(sframe_num)):
                frame_diff = sframe_num[i] - sframe_num[i - 1]
                if frame_diff > 1:
                    # if frame_diff > 1, we have a gap so we need to interpolate
                    interp_bboxes = self.interp(
                        sbboxes_traj[i - 1], sbboxes_traj[i], frame_diff
                    )
                    added_frame_ids = np.arange(sframe_num[i - 1], sframe_num[i], 1)[1:]
                    for j in range(len(interp_bboxes)):
                        self.tracks[key].insert_new_observation(
                            added_frame_ids[j], interp_bboxes[j], sscores[i - 1]
                        )

            # sort tracks again one last time
            post_track = self.tracks[key]
            sorted_bboxes_trajs = sorted(
                zip(post_track.frame_ids, post_track.bboxes_traj), key=itemgetter(0)
            )
            sorted_scores = sorted(
                zip(post_track.frame_ids, post_track.scores), key=itemgetter(0)
            )
            self.tracks[key].frame_ids = [t[0] for t in sorted_bboxes_trajs]
            self.tracks[key].bboxes_traj = [t[1] for t in sorted_bboxes_trajs]
            self.tracks[key].scores = [t[1] for t in sorted_scores]

    def get_sets(self, frame_num):
        """
        fill self.target_stats and self.all_stats with frame with number of frames <= frame_num and
        number of frames > frame_num respectively.
        """
        keys = self.tracks.keys()
        self.target_stats = {}
        self.all_stats = {}
        for key in keys:
            TS = track_stats(self.tracks[key])
            if len(TS) <= frame_num:
                self.target_stats[key] = TS
            self.all_stats[key] = TS

    def append_tracks(self, curr_additions):
        """
        append values of curr_additions to their respectively keys. Both keys and values represent actorIds.
        """
        keys = curr_additions.keys()
        count = 0
        for key in keys:
            for actorID in curr_additions[key]:
                curr_track = self.tracks[actorID]
                count += 1
                self.tracks[key].insert_new_observation(
                    curr_track.frame_ids[0],
                    curr_track.bboxes_traj[0],
                    curr_track.scores[0],
                )
                del self.tracks[actorID]
        print(f"single tracklets removed:{count}")

    def union(self, score_func=0):
        """
        match actorIds based on self.score
        """
        actorIDs = list(self.tracks.keys())
        IDdict = {v: k for k, v in enumerate(actorIDs)}
        n = len(actorIDs)
        self.get_sets(1)

        uf = UnionFind(n)
        for ukey, uvalues in self.target_stats.items():
            min_okey = None
            min_score = float("inf")
            for okey, ovalues in self.all_stats.items():
                # if same actor, continue
                if ukey == okey:
                    continue
                # if target frame_id within test frames, continue
                if ovalues.start <= uvalues.start <= ovalues.end:
                    continue

                if not score_func:
                    curr_score = self.score(uvalues, ovalues)
                else:
                    curr_score = self.score1(uvalues, ovalues)

                if curr_score < min_score:
                    min_score = curr_score
                    min_okey = okey
            if min_score > self.score_thres:
                continue

            uf.union(IDdict[min_okey], IDdict[ukey])

        # get final updated list of connections
        for i in range(n):
            uf.find(i)

        curr_additions = defaultdict(list)
        for i in range(n):
            if i == uf.root[i]:
                continue
            curr_additions[actorIDs[uf.root[i]]].append(actorIDs[i])

        self.append_tracks(curr_additions)
