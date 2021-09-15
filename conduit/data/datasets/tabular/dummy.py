from __future__ import annotations

import numpy as np
import pandas
import pandas as pd
import torch

from conduit.data.datasets.tabular import CdtTabularDataset


class RandomTabularDataset(CdtTabularDataset):
    def __init__(
        self,
        num_disc_features: int = 3,
        num_cont_features: int = 5,
        num_samples: int = 256,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        feats_dict = {}
        feature_groups: list[slice] = []
        for i in range(num_disc_features):
            num_classes = int(rng.integers(low=2, high=10, size=1)[0])
            disc_feat = rng.integers(low=0, high=num_classes, size=num_samples)
            feats_dict[f"disc_{i}"] = pandas.DataFrame(
                self.get_one_hot(disc_feat, num_classes),
                columns=[[f"disc_{i}_{j}" for j in range(num_classes)]],
            )
            prev = len(feature_groups)
            feature_groups += [slice(prev, prev + num_classes)]

        feats = pd.concat(feats_dict.values(), axis=1)
        num_disc_feats = feats.shape[0]

        for i in range(num_cont_features):
            cont_feat = rng.random(num_samples)
            feats[f"cont_{i}"] = cont_feat

        super().__init__(
            x=torch.as_tensor(feats.to_numpy()),
            y=torch.as_tensor(rng.integers(low=0, high=2, size=num_samples)),
            s=None,
            feature_groups=feature_groups,
            disc_indexes=list(range(num_disc_feats)),
            cont_indexes=list(range(num_disc_feats, feats.shape[0])),
        )

    @staticmethod
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])
