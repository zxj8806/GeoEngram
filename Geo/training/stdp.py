import torch

class STDP():
    def __init__(self, a_pos=0.0001, a_neg=-0.01, t_constant=3):
        self.a_pos = a_pos
        self.a_neg = a_neg
        self.t_constant = t_constant

    def setup(self, device, n_neurons):
        self.device = device
        self.neurons = n_neurons

    def per_sample(self, s):
        pass

    def per_time_slice(self, s, k):
        pass

    def train(self, aux, w_latent, spike_latent):
        pre_w = self.a_pos*torch.exp(-aux/self.t_constant)*torch.gt(aux,0).int()
        pos_w = self.a_neg*torch.exp(-aux/self.t_constant)*torch.gt(aux,0).int()
        pre_updates = pre_w*torch.gt((w_latent.T*spike_latent).T, 0).int()
        pos_updates = pos_w*torch.gt(w_latent*spike_latent, 0).int()
        return pre_updates, pos_updates

    def reset(self):
        pass
