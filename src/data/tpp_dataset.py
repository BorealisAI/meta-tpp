# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lightning import LightningModule
from src import constants

logger = logging.getLogger(__name__)

class TPPDataModule(LightningModule):
    def __init__(self, datasets, data_dir, batch_size, num_workers,
                 pin_memory=False, **kwargs):
        super().__init__()
        self.dataset = datasets['dataset']
        self.num_classes = datasets['num_classes']
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = TPPDataset(
            self.data_dir, self.dataset, self.num_classes, mode='train', **self.kwargs)
        self.val_dataset = TPPDataset(
            self.data_dir, self.dataset, self.num_classes, mode='val', **self.kwargs)
        self.test_dataset = TPPDataset(
            self.data_dir, self.dataset, self.num_classes, mode='test', **self.kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=int(self.batch_size/4), num_workers=self.num_workers,
            shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=int(self.batch_size/4), num_workers=self.num_workers,
            shuffle=False, pin_memory=self.pin_memory)



class TPPDataset(Dataset):
    synthetic_data = ['sin']
    real_data = ['so_fold1', 'mooc', 'reddit', 'wiki',
                 'uber_drop', 'taxi_times_jan_feb']

    data_fixed_indices = {
        'so_fold1': [4777, 530]
    }

    def __init__(self, data_dir, dataset, num_classes, mode, **kwargs):
        '''
        data_dir: the root directory where all .npz files are. Default is /shared-data/TPP
        dataset: the name of a dataset
        mode: dataset type - [train, val, test]
        '''
        super(TPPDataset).__init__()
        self.mode = mode

        if dataset in self.synthetic_data:
            data_path = os.path.join(data_dir, 'synthetic', dataset + '.npz')
        elif dataset in self.real_data:
            data_path = os.path.join(data_dir, 'real', dataset + '.npz')
        else:
            logger.error(f'{dataset} is not valid for dataset argument'); exit()

        use_marks = kwargs.get('use_mark', True)
        data_dict = dict(np.load(data_path, allow_pickle=True))
        times = data_dict[constants.TIMES]
        marks = data_dict.get(constants.MARKS, np.ones_like(times))
        masks = data_dict.get(constants.MASKS, np.ones_like(times))
        if not use_marks:
            marks = np.ones_like(times)
        self._num_classes = num_classes

        if dataset not in self.data_fixed_indices:
            (train_size, val_size) = (
                kwargs.get('train_size', 0.6), kwargs.get('val_size', 0.2))
        else:
            train_size, val_size = self.data_fixed_indices[dataset]

        train_rate = kwargs.get('train_rate', 1.0)
        eval_rate = kwargs.get('eval_rate', 1.0)
        num_data = len(times)
        (start_idx, end_idx) = self._get_split_indices(
            num_data, mode=mode, train_size=train_size, val_size=val_size,
            train_rate=train_rate, eval_rate=eval_rate)

        self._times = torch.tensor(
            times[start_idx:end_idx], dtype=torch.float32).unsqueeze(-1)
        self._marks = torch.tensor(
            marks[start_idx:end_idx], dtype=torch.long).unsqueeze(-1)
        self._masks = torch.tensor(
            masks[start_idx:end_idx], dtype=torch.float32).unsqueeze(-1)

        print(f'time shape: {self._times.shape}, marks shape: {self._marks.shape}, masks shape: {self._masks.shape}')


    def _sanity_check(self, time, mask):
        valid_time = time[mask.bool()]
        prev_time = valid_time[0]
        for i in range(1, valid_time.shape[0]):
            curr_time = valid_time[i]
            if curr_time < prev_time:
                logger.error(f'sanity check failed - prev time: {prev_time}, curr time: {curr_time}'); exit()
        logger.info('sanity check passed')

    def _get_split_indices(self, num_data, mode, train_size=0.6, val_size=0.2,
                           train_rate=1.0, eval_rate=1.0):
        if mode == 'train':
            start_idx = 0
            if train_size > 1.0:
                end_idx = int(train_size * train_rate)
            else:
                end_idx = int(num_data * train_size * train_rate)
        elif mode == 'val':
            if val_size > 1.0:
                start_idx = train_size
                end_idx = train_size + val_size
            else:
                start_idx = int(num_data * train_size)
                end_idx = start_idx + int(num_data * val_size * eval_rate)
        elif mode == 'test':
            if train_size > 1.0 and val_size > 1.0:
                start_idx = train_size + val_size
            else:
                start_idx = int(num_data * train_size) + int(num_data * val_size)
            end_idx = start_idx + int((num_data - start_idx) * eval_rate)
        else:
            logger.error(f'Wrong mode {mode} for dataset'); exit()
        return (start_idx, end_idx)

    def __getitem__(self, idx):
        time, mark, mask = self._times[idx], self._marks[idx], self._masks[idx]

        missing_mask = []
        input_dict = {
            constants.TIMES: time,
            constants.MARKS: mark,
            constants.MASKS: mask,
            constants.MISSING_MASKS: missing_mask
        }
        return input_dict

    def __len__(self):
        return self._times.shape[0]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_seq(self):
        return self._times.shape[1]
