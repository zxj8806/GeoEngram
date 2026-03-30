import torch

class NRDP():
    def __init__(self, min_a=0.0, max_a=0.901679611594126, gain_a=0.5428717518563672,
                 min_n=0.0, max_n=0.23001290732040292, gain_n=0.011660312977761912,
                 min_ga=0.0, max_ga=0.7554145024515596, gain_ga=0.3859076787035615,
                 min_gb=0.0, max_gb=0.7954714253083993, gain_gb=0.11032115434326673,
                 time_window=10, gaba_impact=0.01, gaba_rate=0.7):
        self.min_a = min_a
        self.max_a = max_a
        self.gain_a = gain_a
        self.min_n = min_n
        self.max_n = max_n
        self.gain_n = gain_n
        self.min_ga = min_ga
        self.max_ga = max_ga
        self.gain_ga = gain_ga
        self.min_gb = min_gb
        self.max_gb = max_gb
        self.gain_gb = gain_gb
        self.time_window = time_window
        self.gaba_impact = gaba_impact
        self.gaba_rate = gaba_rate

    def setup(self, device, n_neurons):
        self.device = device
        self.n_neurons = n_neurons

    def per_sample(self, s):
        self.firing_state = torch.zeros(self.n_neurons).to(self.device)
        self.a_state = torch.zeros(self.n_neurons).to(self.device)
        self.a_t1 = torch.zeros(self.n_neurons).to(self.device)
        self.n_t1 = torch.zeros(self.n_neurons).to(self.device)
        self.g_t1 = torch.zeros(self.n_neurons).to(self.device)

    def per_time_slice(self, s, k):
        self.gaba_gains = torch.where(torch.rand((self.n_neurons,)) < self.gaba_rate, self.gain_ga, self.gain_gb).to(self.device)
        self.gaba_mins = torch.where(torch.rand((self.n_neurons,)) < self.gaba_rate, self.min_ga, self.min_gb).to(self.device)
        self.gaba_maxes = torch.where(torch.rand((self.n_neurons,)) < self.gaba_rate, self.max_ga, self.max_gb).to(self.device)

    def train(self, aux, w_latent, spike_latent):
        a_t = torch.where(spike_latent > 0,
                          torch.minimum(torch.full_like(self.a_t1, self.max_a), self.a_t1+self.gain_a),
                          torch.where(self.firing_state >= self.time_window,
                                      torch.maximum(torch.full_like(self.a_t1, self.min_a), self.a_t1-(self.gaba_impact*self.gaba_gains)),
                                      self.a_t1))
        self.firing_state[spike_latent < 1] += 1

        a_t = torch.where(a_t >= self.max_a,
                          torch.where(self.a_state <= self.time_window, self.max_a, self.a_t1),
                          self.a_t1)
        self.a_state[a_t >= self.max_a] += 1

        n_t = torch.where(a_t >= self.max_a,
                          torch.where(spike_latent > 0,
                                      torch.minimum(torch.full_like(self.n_t1, self.max_n), self.n_t1+self.gain_n),
                                      torch.maximum(torch.full_like(self.n_t1, self.min_n), self.n_t1-self.gain_n)),
                          self.min_n)

        g_t = torch.where(spike_latent > 0,
                          torch.maximum(self.gaba_mins, self.g_t1-self.gaba_gains),
                          torch.minimum(self.gaba_maxes, self.g_t1+(self.gaba_gains*aux)))

        self.a_t1 = a_t
        self.n_t1 = n_t
        self.g_t1 = g_t

        pre_updates = torch.zeros(self.n_neurons, self.n_neurons, device=self.device)
        pos_updates = ((n_t + a_t) - g_t)*torch.gt(w_latent*spike_latent, 0).int()
        return pre_updates, pos_updates

    def reset(self):
        self.firing_state[self.firing_state > self.time_window] = 0
        self.a_state[self.a_state > self.time_window] = 0
