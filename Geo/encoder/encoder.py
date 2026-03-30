import torch
from abc import ABC, abstractmethod

class Encoder(ABC):
  def __init__(self):
    super().__init__()

  def encode_dataset(self, dataset):
    if dataset.dim() == 2:
      dataset = dataset[:, None, :]

    encoded_data = torch.zeros_like(dataset)

    for i in range(dataset.shape[0]):
      for j in range(dataset.shape[2]):
        encoded_data[i][:, j] = self.encode(dataset[i][:, j])

    return encoded_data

  @abstractmethod
  def encode(self, X_in):
    pass

class Delta(Encoder):
  def __init__(self, threshold=0.1):
    self.threshold = threshold

  def encode(self, sample):
    aux = torch.cat((sample[0].unsqueeze(0), sample))[:-1]
    spikes = torch.ones_like(sample) * (sample - aux >= self.threshold)
    return spikes

class StepForward(Encoder):
  def __init__(self, threshold=0.1):
    super().__init__()
    self.threshold = threshold

  def encode(self, sample):
    spikes = torch.zeros_like(sample, dtype=torch.int8)
    base = sample[0]
    for t in range(1, sample.size(0)):
      if sample[t] >= base + self.threshold:
        spikes[t] = 1
        base += self.threshold
      elif sample[t] <= base - self.threshold:
        spikes[t] = -1
        base -= self.threshold
    return spikes

class PoissonMNIST(Encoder):
  def __init__(self, time_steps=20, p_scale=0.25, seed=None):
    super().__init__()
    self.time_steps = int(time_steps)
    self.p_scale = float(p_scale)
    self.rng = torch.Generator()
    if seed is not None:
      self.rng.manual_seed(int(seed))

  def encode_image_batch(self, images_flat):
    if images_flat.dim() != 2:
      raise ValueError("")
    p = torch.clamp(images_flat, 0.0, 1.0) * self.p_scale
    p = p[:, None, :].expand(-1, self.time_steps, -1)
    spikes = (torch.rand(p.shape, generator=self.rng, device=p.device) < p).to(torch.float32)
    return spikes

  def encode(self, X_in):
    raise NotImplementedError("")
