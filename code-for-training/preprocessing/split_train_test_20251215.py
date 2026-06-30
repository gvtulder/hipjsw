"""
Script used to split the train/validation/test sets at subject level.

Mostly for reference.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import pandas as pd
import numpy as np

SEED = 123

# all available images
df = pd.read_csv("data-src/Baseline/20251022_avail_img_in_dir_Baseline.csv")
df = df.rename(columns=lambda s: s.lower())
# select images with pixel spacing
df = df[df["img_spacing"] > 1e-5]
subject_ids = list(sorted(set(df["coach_id"])))
print(f"Available {len(subject_ids)} subjects")

# all available images
df2 = pd.read_csv("data-src/Follow-up/20251029_avail_img_in_dir_Follow-up_ERGO5.csv")
df2 = df2.rename(columns=lambda s: s.lower())
# select images with pixel spacing
df2 = df2[df2["img_spacing"] > 1e-5]
subject_ids2 = list(sorted(set(df2["coach_id"])))
print(f"Available {len(subject_ids2)} subjects")

subjects = list(sorted(set(subject_ids) | set(subject_ids2)))

# exclude validation subjects
df_exclude = pd.read_csv("data-src/Datasets/20251022_TestSet_JSW_validation.csv")
df_exclude = df_exclude.rename(columns=lambda s: s.lower())
exclude_subject_ids = list(sorted(set(df_exclude["coach_id"])))
print(f"Excluding {len(exclude_subject_ids)} subjects")

# compute remaining
subject_ids = list(sorted(set(subject_ids) - set(exclude_subject_ids)))
print(f"Including {len(subject_ids)} subjects")

rng = np.random.default_rng(seed=SEED)
rng.shuffle(subject_ids)

n_val = int(len(subject_ids) * 0.15)
n_test = int(len(subject_ids) * 0.15)

subjects_per_phase = {
    "test": subject_ids[:n_test],
    "val": subject_ids[n_test:(n_val + n_test)],
    "train": subject_ids[(n_val + n_test):],
}

with open(f"lists/split.20251215.seed{SEED}.txt", "w") as f:
    for phase, subjects in subjects_per_phase.items():
        for subject in subjects:
            f.write(f"{subject},{phase}\n")

