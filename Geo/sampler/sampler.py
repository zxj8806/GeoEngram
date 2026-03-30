import torch
from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, X_in):
        pass

class SpikeCount(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        return spike_activity.sum(axis=1)

class MeanFiringRate(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        return spike_activity.mean(axis=1)

class Binning(Sampler):
    def __init__(self, bin_size=10):
        self.bin_size = bin_size

    def sample(self, spike_activity):
        if not isinstance(spike_activity, torch.Tensor):
            spike_activity = torch.tensor(spike_activity)
        num_time_points = spike_activity.size(1)
        remainder = num_time_points % self.bin_size
        if remainder != 0:
            padding = self.bin_size - remainder
            spike_activity = torch.nn.functional.pad(spike_activity, (0, padding))
        binned_data = spike_activity.unfold(1, self.bin_size, self.bin_size).sum(dim=2)
        flat_binned = binned_data.view(binned_data.size(0), -1)
        return flat_binned

class TemporalBinning(Sampler):
    def __init__(self, bin_size=10):
        self.bin_size = bin_size

    def sample(self, spike_activity):
        if not isinstance(spike_activity, torch.Tensor):
            spike_activity = torch.tensor(spike_activity)
        num_time_points = spike_activity.size(1)
        remainder = num_time_points % self.bin_size
        if remainder != 0:
            padding = self.bin_size - remainder
            spike_activity = torch.nn.functional.pad(spike_activity, (0, 0, 0, padding))
        reshaped = spike_activity.reshape(spike_activity.size(0), -1, self.bin_size, spike_activity.size(2))
        binned_data = reshaped.sum(dim=2)
        flat_binned = binned_data.view(binned_data.size(0), -1)
        return flat_binned

class ISIstats(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        time_steps = torch.arange(spike_activity.shape[1]).unsqueeze(0).unsqueeze(2).expand(spike_activity.shape[0], -1, spike_activity.shape[2])
        spike_indices = time_steps * spike_activity
        spike_indices[spike_indices == 0] = float('inf')
        sorted_spike_indices, _ = torch.sort(spike_indices, dim=1)
        isi = torch.diff(sorted_spike_indices, dim=1)
        isi[isi == float('inf')] = float('nan')
        isi_mean = torch.nanmean(isi, dim=1)
        isi_mean[torch.isnan(isi_mean)] = -1
        return isi_mean

class DeSNN(Sampler):
    def __init__(self, alpha=5, mod=0.8, drift_up=0.8, drift_down=0.01):
        self.alpha = alpha
        self.mod = mod
        self.drift_up = drift_up
        self.drift_down = drift_down

    def sample(self, spike_activity):
        first_spike = (spike_activity != 0).int().argmax(axis=1)
        initial_ranks = self.alpha * (self.mod ** first_spike)
        up = self.drift_up * spike_activity.sum(axis=1)
        down = self.drift_down * (spike_activity.shape[1] - spike_activity.sum(axis=1))
        return initial_ranks + up - down
