import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class Cropper:
    def __init__(self, target_len: int, train: bool):
        self.target_num_samples = target_len
        self.train = train

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        input_num_samples = waveform.shape[-1]
        if input_num_samples == self.target_num_samples:
            return waveform
        if input_num_samples > self.target_num_samples:
            start = (
                torch.randint(0, input_num_samples - self.target_num_samples + 1, (1,)).item()
                if self.train
                else (input_num_samples - self.target_num_samples) // 2
            )
            return waveform[..., start : start + self.target_num_samples]
        padding_needed = self.target_num_samples - input_num_samples
        return F.pad(waveform, (0, padding_needed))


class Libri2sPCM(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        seconds: float = 2.0,
        sr: int = 16000,
        train: bool = True,
    ):
        super().__init__()
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=True)
        self.target_num_samples = int(seconds * sr)
        self.sr = sr
        self.crop = Cropper(self.target_num_samples, train=train)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        waveform, sample_rate, *_ = self.ds[idx]
        if sample_rate != self.sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sr)
        cropped_waveform = self.crop(waveform)
        cropped_waveform = cropped_waveform.clamp(-1.0, 1.0).to(torch.float32)
        return cropped_waveform
