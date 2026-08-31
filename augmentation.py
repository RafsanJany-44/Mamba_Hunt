"""Official RhythmMamba temporal and horizontal-flip augmentation.

This baseline intentionally preserves the public implementation's indexing.
We will test and correct its behavior only after official parity is established.
"""

import random

import numpy as np
import torch

from metrics import calculate_hr


class OfficialAugmentation:
    def __init__(self, fs=30, diff_flag=False):
        self.fs = fs
        self.diff_flag = diff_flag
        self.hr_cache = {}

    def __call__(self, data, labels, recording_ids, chunk_ids):
        batch_size, length, channels, height, width = data.shape
        data_augmented = np.zeros(
            (batch_size, length, channels, height, width)
        )
        labels_augmented = np.zeros((batch_size, length))
        temporal_draws = np.random.random(batch_size)
        flip_draws = np.random.random(batch_size)

        for index_in_batch in range(batch_size):
            cache_key = recording_ids[index_in_batch] + chunk_ids[index_in_batch]
            temporal_draw = temporal_draws[index_in_batch]
            flip_draw = flip_draws[index_in_batch]

            if temporal_draw < 0.5:
                if cache_key in self.hr_cache:
                    ground_truth_hr = self.hr_cache[cache_key]
                else:
                    ground_truth_hr, _ = calculate_hr(
                        labels[index_in_batch],
                        labels[index_in_batch],
                        diff_flag=self.diff_flag,
                        fs=self.fs,
                    )
                    self.hr_cache[cache_key] = ground_truth_hr

                if ground_truth_hr > 90:
                    random_start = random.randint(0, length // 2 - 1)
                    even = torch.arange(0, length, 2)
                    odd = even + 1
                    data_augmented[:, even] = data[
                        :, random_start + even // 2
                    ]
                    labels_augmented[:, even] = labels[
                        :, random_start + even // 2
                    ]
                    data_augmented[:, odd] = (
                        data[:, random_start + odd // 2]
                        + data[:, random_start + odd // 2 + 1]
                    ) / 2
                    labels_augmented[:, odd] = (
                        labels[:, random_start + odd // 2]
                        + labels[:, random_start + odd // 2 + 1]
                    ) / 2
                elif ground_truth_hr < 75:
                    data_augmented[:, : length // 2] = data[:, ::2]
                    labels_augmented[:, : length // 2] = labels[:, ::2]
                    data_augmented[:, length // 2 :] = data_augmented[
                        :, : length // 2
                    ]
                    labels_augmented[:, length // 2 :] = labels_augmented[
                        :, : length // 2
                    ]
                else:
                    data_augmented[index_in_batch] = data[index_in_batch]
                    labels_augmented[index_in_batch] = labels[index_in_batch]
            else:
                data_augmented[index_in_batch] = data[index_in_batch]
                labels_augmented[index_in_batch] = labels[index_in_batch]

        data_augmented = torch.tensor(data_augmented).float()
        labels_augmented = torch.tensor(labels_augmented).float()
        if flip_draw < 0.5:
            data_augmented = torch.flip(data_augmented, dims=[4])
        return data_augmented, labels_augmented
