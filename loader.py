import os
import os.path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class MusicDastset(Dataset):
    _ext_audio = ".wav"
    _ext_txt = ".trans.txt"
    def __init__(self, root: str, train=False) -> None:
        self.root = root
        self.train=train
        classes, class_to_idx = self._find_classes(root)
        self.class_to_idx = class_to_idx
        self.classes = classes
        self.num_classes = len(classes)
        self.melspec_window = 101
        
        melspec_list = []

        for root, _, fnames in sorted(os.walk(self.root)):
            for fname in sorted(fnames):
                if fname.endswith(self._ext_audio):
                    path = os.path.join(root, fname)
                    target_name = path.split("/")[-2]
                    target_idx = self.class_to_idx[target_name]
                    melspec = self.to_melspectrum(path)
                    item = (melspec, target_idx)
                    melspec_list.append(item)
        self.melspec_list = melspec_list

    def to_melspectrum(self, path, window_size=.05, window_stride=.025, 
                          window_type='hamming', normalize=True):
        waveform, sample_rate = soundfile.read(path)
        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)

        # Transformation function
        D = librosa.feature.melspectrogram(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_type)
        melspec, phase = librosa.magphase(D)
        melspec = np.log1p(melspec)
        melspec = torch.FloatTensor(melspec)

        # z-score normalization
        if normalize:
            mean = melspec.mean()
            std = melspec.std()
            if std != 0:
                melspec.add_(-mean)
                melspec.div_(std)
        
        return melspec

    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (melspec, target) where target is class_index of the target class.
        """
        entire_melspec, target = self.melspec_list[index]

        # randomly sample from the melspec
        num_frames = entire_melspec.shape[1]
        rand_frame = np.random.randint(0,num_frames-self.melspec_window-1)
        melspec = entire_melspec[:, rand_frame:rand_frame+self.melspec_window]
        melspec = melspec.unsqueeze(0)
        
        return melspec, target 


    def __len__(self):
        return len(self.melspec_list)

