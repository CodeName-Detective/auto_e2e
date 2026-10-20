"""Registration-time settings every wf_autoe2e task and dynamic must carry.

Image references, the source commit, and Ray storage are read from the
environment when the module is imported. At registration they come from the
buildspec; inside a running ``@dynamic`` they come only from that pod's
environment, so every dynamic passes ``PIPELINE_ENV`` on. Without it, tasks
compiled inside a dynamic fall back to ``:latest`` images and an unknown
commit, which is what made the first smoke run load an image without the
pipeline code.
"""

from __future__ import annotations

import os

_KEYS = (
    "ECR_PREFIX",
    "AUTO_E2E_TRAINING_IMAGE",
    "AUTO_E2E_EVAL_IMAGE",
    "AUTO_E2E_OFFLINE_RL_IMAGE",
    "AUTO_E2E_DATA_PREP_IMAGE",
    "AUTO_E2E_RAY_STORAGE_PATH",
    "AUTO_E2E_GIT_COMMIT",
)

PIPELINE_ENV = {key: os.environ[key] for key in _KEYS if os.environ.get(key)}
