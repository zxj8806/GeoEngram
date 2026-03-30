from Geo.training import STDP
import torch
import torch.nn as nn

class Pipeline():
    def __init__(self, res_model, sampling_method, classifier):
        self.res_model = res_model
        self.sampling_method = sampling_method
        self.classifier = classifier

    def fit(self, X_train, y_train, train=False, learning_rule=STDP(), verbose=False):
        s_act = self.res_model.simulate(X_train, train=train, learning_rule=learning_rule, verbose=verbose)
        state = self.sampling_method.sample(s_act)
        self.classifier.fit(state, y_train)

    def predict(self, X_test):
        s_act = self.res_model.simulate(X_test, train=False, verbose=False)
        state = self.sampling_method.sample(s_act)
        self.state_test = state
        pred = self.classifier.predict(state)
        return pred


class TorchPipeline(nn.Module):
    def __init__(self, res_model, sampling_method, n_classes=10):
        super().__init__()
        self.res_model = res_model
        self.sampling_method = sampling_method
        self.readout = nn.Linear(res_model.n_neurons, n_classes)

    def forward(self, X):
        spk = self.res_model(X)
        state = self.sampling_method.sample(spk)
        logits = self.readout(state)
        return logits
