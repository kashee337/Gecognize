import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from dtw import dtw


class DPMatching:
    def __init__(self, template_dir, threshold=15, normalize=True):
        pathlist = glob.glob(os.path.join(template_dir, "*/*/*.pickle"))
        df = pd.DataFrame({"path": pathlist})
        df["group"] = df["path"].agg(lambda t: t.split("/")[-2])
        df["template"] = df["path"].agg(lambda t: t.split("/")[-3])
        self.template_dict = DPMatching.gen_template(df, normalize)
        self.threshold = threshold
        self.normalize = normalize

    @staticmethod
    def gen_template(df, normalize):
        template_dict = defaultdict(list)
        groups = df["group"].unique()
        for group in groups:
            _df = df.query("group==@group")
            data = DPMatching.extract_data(_df["path"], normalize)
            name = _df["template"].values[0]
            template_dict[name].append(data)
        return template_dict

    @staticmethod
    def extract_data(pathlist, normalize):
        data = []
        for path in sorted(pathlist):
            with open(path, "rb") as f:
                tmp = pickle.load(f)
            data.append([tmp.landmark[8].x, tmp.landmark[8].y])

        if normalize:
            data = DPMatching.normalize_traj(data)
        else:
            data = np.asarray(data)
        return data

    @staticmethod
    def normalize_traj(data):
        data = np.asarray(data)
        min_x = np.min(data.T[0])
        max_x = np.max(data.T[0])
        min_y = np.min(data.T[1])
        max_y = np.max(data.T[1])
        data -= np.array([min_x, min_y])
        data /= np.array([max_x - min_x, max_y - min_y])
        return data

    def __call__(self, s1):

        if self.normalize:
            s1 = DPMatching.normalize_traj(s1)

        score = defaultdict(list)
        for k, s2_list in self.template_dict.items():
            for s2 in s2_list:
                dp = dtw(s1, s2)
                score[k].append(np.sqrt(dp[-1][-1]))

        self.score = sorted(score.items(), key=lambda x: np.median(x[1]))
        inf_res = self.score[0]
        if np.median(self.score[0][1]) <= self.threshold:
            return inf_res[0], np.median(self.score[0][1])
        else:
            return "?", "Unfefined"
