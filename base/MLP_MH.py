import torch
import torch.nn as nn
#multi-head MLP
class MLP_MH(nn.Module):
    def __init__(self, input, output, hidden_layers_features, output_type=None):
        '''
        :param input:
        :param output:
        :param hidden_layers_features:
        :param output_type:  None-> value
                             prob-> softmax
                             gauss-> mu, sigma
                             multi_discrete -> [softmax, softmax,...]
        '''
        super(MLP_MH, self).__init__()
        self.input = input
        self.output = output
        self.hidden_layers_features = hidden_layers_features
        self.output_type = output_type

        self.layers = []
        self.layers.append(nn.Linear(self.input, self.hidden_layers_features[0]))
        for i in range(len(hidden_layers_features) - 1):
            self.layers.append(nn.Linear(self.hidden_layers_features[i], self.hidden_layers_features[i + 1]))
        self.layers = nn.ModuleList(self.layers)

        if self.output_type == "gauss":
            self.mu = nn.Linear(self.hidden_layers_features[-1], self.output)
            self.sigma = nn.Linear(self.hidden_layers_features[-1], self.output)
        elif self.output_type == "multi_discrete":
            self.out = nn.ModuleList([nn.Linear(self.hidden_layers_features[-1], n_output) for n_output in self.output])
        else:
            self.out = nn.Linear(self.hidden_layers_features[-1], self.output)

    def forward(self, x):
        assert type(x) is torch.Tensor, "net forward input type error"
        for i, layer in enumerate(self.layers):
            x = nn.functional.tanh(layer(x))


        if self.output_type == "gauss":
            mu = nn.functional.tanh(self.mu(x))
            sigma = nn.functional.softplus(self.sigma(x))
            return mu, sigma
        elif self.output_type == "prob":
            hidden_prob = x.detach()
            x = nn.functional.softmax(self.out(x))
            #return x, hidden_prob   #hidden prob
            return x, x.detach()   #trub prob
        elif self.output_type == "multi_discrete":
            hidden_prob = x.detach()
            x = [nn.functional.softmax(out(x)) for out in self.out]
            trub_prob = [temp.detach() for temp in x]
            # return x, hidden_prob   #hidden prob
            return x, trub_prob  # trub prob
        else:
            x = self.out(x)
            return x

if __name__ == "__main__":
    mlp = MLP_MH(3, [3, 3], [2, 2], output_type="multi_discrete")
    a = torch.Tensor([1, 2, 3])
    b = mlp(a)
    print(b)
    pass