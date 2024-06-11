from copy import deepcopy
from base.MLP_MH import MLP_MH
from utils.torch_tool import soft_update
from collections import OrderedDict
import numpy as np
import types
import torch
from memory_profiler import profile
class Opponent_Model_MH(object):
    def __init__(self, args, conf, name, device=None):
        self.model = MLP_MH(input=conf["n_state"],
                            output=conf["n_opponent_action"],
                            hidden_layers_features=conf["opponent_model_hidden_layers"],
                            output_type="multi_discrete")
        self.args = args
        self.conf = conf
        self.name = "MH_" + name
        self.device = device

        if device:
            self.model.to(device)

    def change_device(self, device):
        self.device = device
        self.model.to(device)

    def get_parameter(self, *args):
        return [p.clone() for p in self.model.parameters()]

    def set_parameter(self, new_parameter):
        if isinstance(new_parameter, types.GeneratorType):
            new_parameter = list(new_parameter)
        for target_param, param in zip(list(self.model.parameters()), new_parameter):
            target_param.data.copy_(param.data)

    def get_action_prob(self, state, param=None):
        '''
        :param state: np.ndarry or torch.Tensor shape is (n_batch, n_state)
        :param param: models's param: list or np.ndarry  shape is (1)
        :return: action_prob: torch.Tensor shape is (n_batch, n_action_prob)
                 hidden_prob: torch.Tensor shape is (n_batch, n_hidden_prob)
        '''
        assert ((type(state) is np.ndarray) or (type(state) is torch.Tensor)), "get_action_prob input type error"
        if (type(state) is np.ndarray):
            state = torch.Tensor(state)
        if self.device:
            state = state.to(self.device)
        if param is not None:
            self.set_parameter(param)
        action_prob, hidden_prob = self.model(state)
        return action_prob, hidden_prob

    def learn(self, data, param, lr, l_times):
        '''
        :param data: dict("state":  ndarray or torch.Tensor float [n_batch, n_state],
                          "action": ndarray or torch.Tensor int [n_batch, 1])
        :param param: old_params: [1]
        :param lr: learning rate: float
        :param l_times: train times: int
        :return: new_param: [1]
                 loss: float
        '''
        #assert len(data["state"].shape) == 2 and len(data["action"].shape) == 2, "learn data shape error"
        if type(data["state"]) is np.ndarray:
            state = torch.Tensor(data["state"])
        else:
            state = data["state"]
        if type(data["action"][0]) is np.ndarray:
            action_target = [torch.LongTensor(data["action"][i]) for i in range(len(data["action"]))]
        else:
            action_target = data["action"]
        if self.device:
            state = state.to(self.device)
            action_target = [action_target[i].to(self.device) for i in range(len(action_target))]
        self.set_parameter(param)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(l_times):
            optimizer.zero_grad()
            action_eval, _ = self.model(state)
            entropy = [torch.distributions.categorical.Categorical(action_eval[i]).entropy().mean() for i in range(len(action_eval))]
            loss = torch.stack([loss_fn(action_eval[0], action_target[0].squeeze(1)) -1 * entropy[0] for i in range(len(action_eval))]).mean()
            loss.backward()
            optimizer.step()
        return self.get_parameter(), float(loss)

    def eval(self, fn_name):
        return eval("self." + fn_name)
class OM_MH_Buffer(object):
    def __init__(self, args, conf, device=None):
        self.args = args
        self.conf = conf
        self.device = device
        self.output_dim = len(conf["n_opponent_action"])

        self.obs = torch.zeros((conf["opponent_model_memory_size"], conf["n_state"]), dtype=torch.float32, device=device)
        self.act = [torch.zeros((conf["opponent_model_memory_size"], 1), dtype=int, device=device) for _ in range(self.output_dim)]
        self.next_idx, self.size, self.max_size = 0, 0, conf["opponent_model_memory_size"]

    def store_memory(self, state, action):
        '''
        :param state: np.ndarray or torch.Tensor shape is (n_batch, n_state) float
        :param action: np.ndarray or torch.Tensor shape is (n_batch, n_state) int
        :return: None
        '''
        #assert len(state.shape) == 2 and len(action.shape) == 2, "store_memory data shape error"
        #assert state.shape[0] == action.shape[0], "store_memory data shape error"
        assert state.shape[1] == self.obs.shape[1], "store_memory state shape error"
        #assert action.shape[1] == 1, "store_memory action shape error"
        if type(state) is np.ndarray:
            state = torch.Tensor(state).float()
        if type(action[0]) is np.ndarray:
            action = [torch.LongTensor(action[i]) for i in range(self.output_dim)]
        if self.device:
            state = state.to(self.device)
            action = [action[i].to(self.device) for i in range(self.output_dim)]
        n_batch = state.shape[0]
        if self.next_idx + n_batch < self.max_size:
            self.obs[self.next_idx:self.next_idx+n_batch] = state
            for i in range(self.output_dim):
                self.act[i][self.next_idx:self.next_idx + n_batch] = action[i]
            self.next_idx = self.next_idx + n_batch
            self.size = max(self.size, self.next_idx)
        else:
            temp_idx = self.max_size - self.next_idx
            self.obs[self.next_idx:self.max_size] = state[:temp_idx]
            for i in range(self.output_dim):
                self.act[i][self.next_idx:self.max_size] = action[i][:temp_idx]
            self.next_idx = n_batch - (self.max_size - self.next_idx)
            self.obs[:self.next_idx, :] = state[temp_idx:, :]
            for i in range(self.output_dim):
                self.act[i][:self.next_idx, :] = action[i][temp_idx:, :]
            self.size = self.max_size

    def clear_memory(self):
        self.next_idx = 0
        self.size = 0

    def get_batch(self, batch_size):
        idxes = torch.randperm(self.size, device=self.device)[:min(batch_size, self.size)]
        state = self.obs[idxes]
        action = [self.act[i][idxes] for i in range(self.output_dim)]
        return dict({"state": state, "action": action})

    def eval(self, fn_name):
        return eval("self." + fn_name)


if __name__ == "__main__":
    import argparse
    import numpy as np
    from config.simple_tag_conf import player2_conf

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--num_om_layers", type=int, default=2, help="Number of trajectories")

    args = parser.parse_args()

    om_mh = Opponent_Model_MH(args, player2_conf, "MBAM")
    param = om_mh.get_parameter()
    #om_mh.set_parameter(param)
    state = np.random.random((5,20))
    action_prob, hidden_prob = om_mh.get_action_prob(state=state, param=param)
    pass

    ombf = OM_MH_Buffer(args, player2_conf, "cpu")
    for i in range(100):
        BATCH_SIZE = np.random.randint(100)
        state = np.random.random((BATCH_SIZE, 20))
        action = [np.random.randint(0, 5, (BATCH_SIZE, 1)) for j in range(3)]
        ombf.store_memory(state, action)
        data = ombf.get_batch(100)
        param, loss = om_mh.learn(data=data, param=param, lr=0.001, l_times=5)
        print(loss)
    pass