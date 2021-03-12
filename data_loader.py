import os
import os.path

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile 
import pandas as pd


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

class BooksDataset(Dataset):
    """Create a Dataset for LibriSpeech.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"
    labels = "LabelsFinal.csv"
    URL = "train-clean-100"
    FOLDER_IN_ARCHIVE = "LibriSpeech"

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)
        df = pd.read_csv(os.path.join(root,self.labels))
        classes = list(df['Genre'].unique())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.num_classes = len(classes)
        
        spect_list = []

        for root, _, fnames in sorted(os.walk(self._path)):
            for fname in sorted(fnames):
                if fname.endswith(self._ext_audio):
                    path = os.path.join(root, fname)
                    chapter_id = int(fname.split('-')[1])
                    target = list(df.loc[df['ID'] == chapter_id,'Genre'])
                    item = (path, class_to_idx[target[0]])
                    spect_list.append(item)
        self.spect_list = spect_list
    
    def load_librispeech_item(self,fileid, path, ext_audio, ext_txt, window_size=.05, window_stride=.05, 
                          window_type='hamming', normalize=True, max_len=101):
    #speaker_id, chapter_id, utterance_id = fileid.split("-")

    #file_text = speaker_id + "-" + chapter_id + ext_txt
    #file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    #fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    #file_audio = fileid_audio + ext_audio
    #file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
        waveform, sample_rate = torchaudio.load(fileid)
    
        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)

    
    
        D = librosa.feature.melspectrogram(waveform.numpy().squeeze(), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_type)
        spect, phase = librosa.magphase(D)



    # S = log(S+1)
        spect = np.log1p(spect)

        if spect.shape[1] < max_len:
            pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
            spect = np.hstack((spect, pad))
        elif spect.shape[1] > max_len:
            spect = spect[:, :max_len]
        spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
        spect = torch.FloatTensor(spect)

    # z-score normalization
        if normalize:
            mean = spect.mean()
            std = spect.std()
            if std != 0:
                spect.add_(-mean)
                spect.div_(std)
    
            return spect


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
            tuple: (spect, target) where target is class_index of the target class.
        """
        fileid, target = self.spect_list[index]
        spect = self.load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)
        
        return spect, target 


    def __len__(self):
        return len(self.spect_list)

