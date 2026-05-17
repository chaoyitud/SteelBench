"""TabPFN v3 — mediumdata checkpoint variant.

Uses the ``tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt`` checkpoint
(trained on medium-sized datasets, released 2026-04-17) instead of the
default v3 checkpoint.  Everything else is identical to TabPFNV3Method.
"""

import os
from TALENT.model.methods.tabpfn_v3 import TabPFNV3Method

_CKPT_NAME = "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"
_CKPT_PATH = os.path.expanduser(os.path.join("~", ".cache", "tabpfn", _CKPT_NAME))


class TabPFNV3MediumMethod(TabPFNV3Method):
    """TabPFN v3 with the April-2026 mediumdata checkpoint."""

    def construct_model(self, cat_indices=[]):
        from tabpfn import TabPFNRegressor, TabPFNClassifier

        if self.is_regression:
            self.model = TabPFNRegressor(
                model_path=_CKPT_PATH,
                device=self.args.device,
                random_state=self.args.seed,
                n_estimators=8,
                ignore_pretraining_limits=True,
                categorical_features_indices=cat_indices if cat_indices else None,
            )
        else:
            self.model = TabPFNClassifier(
                model_path=_CKPT_PATH,
                device=self.args.device,
                random_state=self.args.seed,
                n_estimators=8,
                ignore_pretraining_limits=True,
                categorical_features_indices=cat_indices if cat_indices else None,
            )
