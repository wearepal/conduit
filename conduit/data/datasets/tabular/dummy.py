from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from conduit.data.datasets.tabular import CdtTabularDataset

__all__ = ["RandomTabularDataset"]


class RandomTabularDataset(CdtTabularDataset):
    def __init__(
        self,
        num_disc_features: int = 3,
        num_cont_features: int = 5,
        num_samples: int = 256,
        seed: int = 0,
        s_card: Optional[int] = None,
        y_card: Optional[int] = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        feats_dict = {}
        feature_groups: List[slice] = []
        prev = 0
        for i in range(num_disc_features):
            num_classes = int(rng.integers(low=2, high=10, size=1)[0])
            disc_feat = pd.Series(rng.integers(low=0, high=num_classes, size=num_samples))
            ohe_feats = pd.get_dummies(disc_feat)
            feats_dict[f"disc_{i}"] = ohe_feats
            feature_groups += [slice(prev, prev + ohe_feats.shape[1])]
            prev += ohe_feats.shape[1]

        feats = pd.concat(feats_dict.values(), axis=1)
        num_disc_feats = feats.shape[1]

        for i in range(num_cont_features):
            cont_feat = rng.random(num_samples)
            feats[f"cont_{i}"] = cont_feat

        if y_card is None:
            y = None
        else:
            y = torch.as_tensor(rng.integers(low=0, high=y_card, size=num_samples))

        if s_card is None:
            s = None
        else:
            s = torch.as_tensor(rng.integers(low=0, high=s_card, size=num_samples))

        super().__init__(
            x=torch.as_tensor(feats.to_numpy()),
            y=y,
            s=s,
            feature_groups=feature_groups,
            disc_indexes=list(range(num_disc_feats)),
            cont_indexes=list(range(num_disc_feats, feats.shape[1])),
        )
