import sys
sys.path.append("/home/lenovo/文档/CodeWorkspace/RL")
from DRL_MARL_homework.MBAM.baselines.PPO import PPO, PPO_Buffer
from DRL_MARL_homework.MBAM.policy.Opponent_Model import Opponent_Model, OM_Buffer
from DRL_MARL_homework.MBAM.utils.datatype_transform import dcn
from DRL_MARL_homework.MBAM.utils.rl_utils import discount_cumsum
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import types
import numpy as np
from collections import OrderedDict

class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in Duan et al., 2016
    Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
    "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
    Args:
        input_size (int): Input size from environment
        args (argparse): Python argparse that contains arguments
        reg_coeff (float): Regularization coefficient. Default: 1e-5
    Reference:
        https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/baseline.py
    """

    def __init__(self, input_size, args, conf, device, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()

        self.input_size = input_size
        self.args = args
        self.conf = conf
        self.device = device
        self._reg_coeff = reg_coeff

        self.weight = nn.Parameter(torch.Tensor(self.feature_size, ), requires_grad=False).to(device=self.device)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.weight.device)

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, obs):
        batch_size, sequence_length, _ = obs.shape
        #assert sequence_length == self.args.eps_max_step, "Length should equal to episodic horizon"

        ones = torch.ones((sequence_length, batch_size, 1)).to(device=self.device)
        obs = obs.clone().transpose(0, 1).to(device=self.device)
        time_step = (torch.arange(sequence_length).view(-1, 1, 1).to(device=self.device) * ones / 100.0).to(device=self.device)

        return torch.cat([
            obs,
            obs ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)

    def fit(self, obs, return_):
        # featmat.shape: sequence_length * batch_size x feature_size
        featmat = self._feature(obs).view(-1, self.feature_size)

        # returns.shape: sequence_length * batch_size x 1
        returns = return_.view(-1, 1)

        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        for _ in range(5):
            try:
                coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff * self._eye)
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError(
                'Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.weight.copy_(coeffs.flatten())

    def get_return(self, reward):
        """Compute episodic return given trajectory

        Args:
            reward (list): Contains rewards across trajectories for specific agent
            args (argparse): Python argparse that contains arguments
        Returns:
            return_ (torch.Tensor): Episodic return with shape: (batch, ep_horizon)
        """
        # reward = torch.stack(reward, dim=1)
        # assert reward.shape == (args.traj_batch_size, args.ep_horizon), \
        #    "Shape must be: (batch, ep_horizon)"

        R, return_ = 0., []
        for timestep in reversed(range(self.args.eps_max_step)):
            R = reward[:, timestep] + self.conf["gamma"] * R
            return_.insert(0, R)
        return_ = torch.stack(return_, dim=1)

        return return_
    def forward(self, obs, reward_to_go):
        # Fit linear feature baseline
        '''
        :param obs: np.ndarray (eps_max_step, num_trj, n_state)
        :param reward_to_go: torch.Tensor (num_trj, max_step)
        :return:
        '''
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(obs).float().to(self.device)
        else:
            obs = obs.to(self.device)
        #return_ = self.get_return(reward)
        self.fit(obs, reward_to_go)

        # Return value
        features = self._feature(obs)
        value = torch.mv(features.view(-1, self.feature_size), self.weight)

        return value.view(features.shape[:2]).transpose(0, 1)

class Meta_mapg(PPO):
    def __init__(self, args, conf, name, logger, agent_idx, actor_rnn, device=None):
        super(PPO, self).__init__(a_n_state=conf["n_state"],
                                  v_n_state=conf["n_state"],
                                  n_action=conf["n_action"],
                                  a_hidden_layers=conf["a_hidden_layers"],
                                  v_hidden_layers=conf["v_hidden_layers"],
                                  actor_rnn=actor_rnn,
                                  args=args,
                                  conf=conf,
                                  name="Meta_mapg_" + name,
                                  logger=logger)
        self.agent_idx = agent_idx      #agent_index in env
        self.oppo_model = Opponent_Model(args, conf, self.name, device)
        self.om_buffer = OM_Buffer(args, conf, device)
        self.device = device
        if device is not None:
            self.change_device(device)
        self.linear_baseline = LinearFeatureBaseline(input_size=self.conf["n_state"], args=self.args, conf=self.conf, device=self.device)
        self._set_dynamic_lr()

    def change_device(self, device):
        super(Meta_mapg, self).change_device(device)
        self.oppo_model.change_device(device)

    def _set_dynamic_lr(self):
        initial_lr = np.array([self.conf["actor_lr_inner"] for _ in range(self.conf["chain_horizon"])])
        self.dynamic_lr = torch.nn.Parameter(torch.from_numpy(initial_lr).float(), requires_grad=True)
        self.dynamic_lr_optimizer = torch.optim.Adam((self.dynamic_lr,), lr=self.conf["actor_lr_outer"])

    def get_a_parameter(self, *args):
        return [p.clone() for p in self.a_net.parameters()]

    def set_parameter(self, new_parameter):
        if isinstance(new_parameter, types.GeneratorType):
            new_parameter = list(new_parameter)
        for target_param, param in zip(list(self.a_net.parameters()), new_parameter):
            target_param.data.copy_(param.data)

    def choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
        '''
        :param state: np.ndarry  shape is (n_batch, n_state)
        :param greedy:
        :param kwargs: if oppo_hidden_prob == None  return deal mixed_opponent action
                       else return deal assigned opponent action
        :return: action, np.ndarray int32 (n_batch, 1)
                 logp_a, np.ndarray float (n_batch, 1)
                 entropy, np.ndarray float (n_batch, 1)
                 value, np.ndarray float (n_batch, 1)
                 action_prob, np.ndarray float (n_batch, n_action)
                 hidden_prob, np.ndarray float (n_batch, n_hidden_prob), this is actor network number of latest layer'cell
                 hidden_state, np.ndarray float (n_batch, n_hidden_state), is None if not actor_rnn
        '''
        assert type(state) is np.ndarray or type(state) is torch.Tensor, "choose_action input type error"
        if type(state) is np.ndarray:
            state = torch.Tensor(state).view(-1, self.conf["n_state"]).to(device=self.device)
        with torch.no_grad():
            oppo_action_prob, oppo_hidden_prob = self.oppo_model.get_action_prob(state)
            oppo_pi = torch.distributions.Categorical(oppo_action_prob)
            oppo_action = oppo_pi.sample()
            oppo_logp_a = dcn(oppo_pi.log_prob(oppo_action))
        if self.actor_rnn:
            return list((self._choose_action(state, greedy, hidden_state=hidden_state))) + [oppo_logp_a]
        return list((self._choose_action(state, greedy, hidden_state=None))) + [oppo_logp_a]

    def _choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
        '''
        :param state: np.ndarry or torch.Tensor shape is (n_batch, n_state)
        :param greedy:
        :return: action, np.ndarray int32 (n_batch, 1) 1 dim
                 logp_a, np.ndarray float (n_batch, 1) 1 dim
                 entropy, np.ndarray float (n_batch, 1) 1 dim
                 value, np.ndarray float (n_batch, 1) 1 dim
                 action_prob, np.ndarray float (n_batch, n_action)
                 hidden_prob, np.ndarray float (n_batch, n_hidden_prob), this is actor network number of latest layer'cell
                 hidden_state, np.ndarray float (n_batch, n_hidden_state), is None if not actor_rnn
        '''
        assert (type(state) is np.ndarray) or (type(state) is torch.Tensor), "choose_action input type error"
        if type(state) is np.ndarray:
            state = state.reshape(-1, self.conf["n_state"])
            a_state = torch.Tensor(state).to(device=self.device)
            v_state = torch.Tensor(state).to(device=self.device)
        else:
            a_state = state.clone()
            v_state = state
        if self.device:
            a_state = a_state.to(self.device)
            v_state = v_state.to(self.device)
        if oppo_hidden_prob is not None:
            if type(oppo_hidden_prob) is np.ndarray:
                oppo_hidden_prob = torch.Tensor(oppo_hidden_prob).to(device=self.device)
            oppo_hidden_prob = oppo_hidden_prob.view((-1, self.conf["n_opponent_action"] if self.args.true_prob else
            self.conf["opponent_model_hidden_layers"][-1]))
            a_state = torch.cat([a_state, oppo_hidden_prob], dim=1)
        if hidden_state is not None:
            if type(hidden_state) is np.ndarray:
                hidden_state = torch.Tensor(hidden_state).to(device=self.device)
            hidden_state = hidden_state.view((-1, self.conf["a_hidden_layers"][0]))
        value = self.v_net(v_state)
        if self.actor_rnn:
            action_prob, hidden_prob, hidden_state = self.a_net(a_state, hidden_state)
        else:
            action_prob, hidden_prob = self.a_net(a_state)

        if greedy:
            pi = Categorical(action_prob)
            _, action = torch.max(action_prob, dim=1)
            logp_a = pi.log_prob(action)
            entropy = pi.entropy()
        else:
            pi = torch.distributions.Categorical(action_prob)
            action = pi.sample()
            logp_a = pi.log_prob(action)
            entropy = pi.entropy()
        return (dcn(action).astype(np.int32),
                logp_a,
                dcn(entropy),
                dcn(value),
                dcn(action_prob),
                dcn(hidden_prob),
                dcn(hidden_state) if self.actor_rnn else None)

    def get_dice_loss(self, logp_a, reward_to_go, value, mask, oppo_logp_a=None, is_train=False):
        """Compute DiCE loss
        In our code, we use DiCE in the inner loop to be able to keep the dependency in the
        adapted parameters. This is required in order to compute the opponent shaping term.
        Args:
            logprobs (list): Contains log probability of all agents
                             shape is (2, num_trj, max_step)
            reward (list): Contains rewards across trajectories for specific agent
                             shape is (num_trj, max_step)
            value (tensor): Contains value for advantage computed via linear baseline
                             shape is (num_trj, max_step)
            mask (tensor): [1 1 1 ... 0 0]
                             shape is (num_trj, max_step)
            args (argparse): Python argparse that contains arguments
            i_agent (int): Agent to compute DiCE loss for
            is_train (bool): Flag to identify whether in meta-train or not
        Returns:
            dice loss (tensor): DiCE loss with baseline reduction
        References:
            https://github.com/alshedivat/lola/blob/master/lola_dice/rpg.py
            https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
        """

        def magic_box(x):
            """DiCE operation that saves computation graph inside tensor
            See ``Implementation of DiCE'' section in the DiCE Paper for details
            Args:
                x (tensor): Input tensor
            Returns:
                1 (tensor): Tensor that has computation graph saved
            References:
                https://github.com/alshedivat/lola/blob/master/lola_dice/rpg.py
                https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
            """
            return torch.exp(x - x.detach())

        # Get discounted_reward
        # reward = torch.stack(reward, dim=1)
        cum_discount = torch.cumprod(self.conf["gamma"] * torch.ones(*reward_to_go.size()), dim=1).to(self.device) / self.conf["gamma"]
        #discounted_reward = reward * cum_discount

        # Compute stochastic nodes involved in reward dependencies
        if args.meta_oppo_modeling and is_train:
            logprob_sum, stochastic_nodes = 0., 0.
            for logprob in [logp_a, oppo_logp_a]:
                # logprob = torch.stack(logprob, dim=1)
                logprob_sum += logprob
                stochastic_nodes += logprob
            dependencies = torch.cumsum(logprob_sum, dim=1)
        else:
            # logprob = logprobs[i_agent]
            # logprob = torch.stack(logprobs[i_agent], dim=1)
            dependencies = torch.cumsum(logp_a, dim=1)
            stochastic_nodes = logp_a

        # Get DiCE loss
        dice_loss = torch.mean(torch.sum(magic_box(dependencies) * reward_to_go * mask, dim=1))
        # dice_loss = torch.mean(torch.sum(magic_box(dependencies) * reward_to_go * mask, dim=1) / torch.sum(mask, dim=1))

        # Apply variance_reduction if value is provided
        baseline_term = 0.
        if value is not None:
            discounted_value = value.detach() * cum_discount * mask
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_value, dim=1))

        return -(dice_loss + baseline_term)

    def inner_update(self, data, phi, i_joint, is_train, iteration, no_log=True):
        '''
        :param data: data contains state, action, reward_to_go, advantage, logp_a, hidden_state, mask
        :keyword state: torch.Tensor shape is (n_trj, trj_length, n_state) float
                 action: torch.Tensor shape is (n_trj, trj_length, 1) int
                 reward_to_go: torch.Tensor shape is (n_trj, trj_length, 1) float
                 logp_a: torch.Tensor shape is (n_trj, trj_length, 1) float
                 advantage: torch.Tensor shape is (n_trj, trj_length, 1) float
                 hidden_state: torch.Tensor shape is (n_trj, trj_length, n_hidden_state) float
                 oppo_logp_a: torch.Tensor shape is (n_trj, trj_length, 1) float
                 mask: action: torch.Tensor shape is (n_trj, trj_length, 1) bool
        :return:
        '''
        if i_joint == self.conf["chain_horizon"]:
            return None
        #obs, logprobs, _, _, rewards = memory.sample()

        state = data["state"].to(device=self.device)
        logp_a = data["logp_a"].squeeze(2).to(device=self.device)
        oppo_logp_a = data["oppo_logp_a"].squeeze(2).to(device=self.device)
        reward_to_go = data["reward_to_go"].squeeze(2).to(device=self.device)
        mask = data["mask"].squeeze(2).to(device=self.device)
        # Compute value for baseline
        #reward = rewards[self.i_agent]
        value = self.linear_baseline(state, reward_to_go)

        # Compute DiCE loss
        actor_loss = self.get_dice_loss(logp_a, reward_to_go, value, mask, oppo_logp_a, is_train)

        #phi = self.get_a_parameter()
        # Get adapted parameters
        actor_grad = torch.autograd.grad(actor_loss, self.a_net.parameters(), create_graph=is_train)
        # actor_grad = torch.autograd.grad(actor_loss, phi, create_graph=is_train,)

        # if hasattr(self, 'is_tabular_policy'):
        #     phi = actor - 1. * actor_grad[0]
        # else:
        new_phi = OrderedDict()
        for (name, param), grad in zip(self.a_net.named_parameters(), actor_grad):
            new_phi[name] = param - self.dynamic_lr[i_joint] * grad
        #self.set_parameter(new_parameter=new_phi)
        return new_phi

    def compute_outer_loss(self, datum, iteration, no_log=False):
        '''
        :param datum: data list, length is chain_horizon + 1
                      data contains state, action, reward_to_go, advantage, logp_a, hidden_state, mask
        :keyword state: torch.Tensor shape is (n_trj, trj_length, n_state) float
                 action: torch.Tensor shape is (n_trj, trj_length, 1) int
                 reward_to_go: torch.Tensor shape is (n_trj, trj_length, 1) float
                 logp_a: torch.Tensor shape is (n_trj, trj_length, 1) float
                 advantage: torch.Tensor shape is (n_trj, trj_length, 1) float
                 hidden_state: torch.Tensor shape is (n_trj, trj_length, n_hidden_state) float
                 mask: action: torch.Tensor shape is (n_trj, trj_length, 1) bool
        :param iteration:
        '''
        '''a_loss'''
        # Get advantage
        advantage = datum[-1]["advantage"]

        # Get current policy loss
        logp_a = datum[0]["logp_a"]
        mask = datum[0]["mask"]
        #current_policy_loss = torch.mean(torch.sum((torch.sum((logp_a * advantage * mask), dim=1)), dim=0) / torch.sum(mask, dim=0))
        # current_policy_loss = torch.mean(torch.sum(logp_a * advantage * mask, dim=1))
        current_policy_loss = torch.mean(torch.mean(torch.sum(logp_a * advantage * mask, dim=1)) / torch.sum(mask, dim=1))

        # Get own learning loss
        own_learning_loss = 0.
        for i, data in enumerate(datum[1:]):
            logp_a = data["logp_a"]
            temp_mask = data["mask"] * datum[0]["mask"]
            #own_learning_loss += torch.mean(torch.sum((torch.sum((logp_a * advantage * mask), dim=1)), dim=0) / torch.sum(mask, dim=0))
            # own_learning_loss += torch.mean(torch.sum(logp_a * advantage * temp_mask, dim=1))
            own_learning_loss += torch.mean(torch.mean(torch.sum(logp_a * advantage * temp_mask, dim=1)) / torch.sum(mask, dim=1))

        # Get opponent learning loss
        opponent_learning_loss = 0.
        if args.meta_oppo_modeling:
            for i, data in enumerate(datum[1:]):
                oppo_logp_a = data["oppo_logp_a"]
                temp_mask = data["mask"] * datum[0]["mask"]
                #own_learning_loss += torch.mean(torch.sum((torch.sum((oppo_logp_a * advantage * mask), dim=1)), dim=0) / torch.sum(mask, dim=0))
                # own_learning_loss += torch.mean(torch.sum(oppo_logp_a * advantage * temp_mask, dim=1))
                own_learning_loss += torch.mean(torch.mean(torch.sum(oppo_logp_a * advantage * temp_mask, dim=1)) / torch.sum(mask, dim=1))

        a_loss = -(current_policy_loss + own_learning_loss + opponent_learning_loss)

        '''v_loss'''
        state = datum[-1]["state"]
        reward_to_go = datum[-1]["reward_to_go"]
        temp_mask = datum[-1]["mask"]
        v_loss = torch.mean(torch.sum(((self.v_net(state) * temp_mask - reward_to_go) ** 2), dim=1) / torch.sum(mask, dim=1))
        # v_loss = torch.mean(torch.sum((torch.sum(((self.v_net(state) - reward_to_go) ** 2), dim=1)), dim=0) / torch.sum(mask, dim=0))

        # log
        pass

        return a_loss, v_loss

    def outer_update(self, loss, iteration, update_type):
        if update_type == "actor":
            network = self.a_net
            optimizer = self.a_optimizer
            #key, tb_key = "/actor_grad", "rank" + str(self.rank) + "/outer/actor_loss_avg"
        elif update_type == "dynamic_lr":
            network = self.dynamic_lr
            optimizer = self.dynamic_lr_optimizer
            #key, tb_key = "/dynamic_lr_grad", None
        elif update_type == "value":
            network = self.v_net
            optimizer = self.v_optimizer
            #key, tb_key = "/value_grad", "rank" + str(self.rank) + "/outer/value_loss_avg"
        else:
            raise ValueError()

        optimizer.zero_grad()
        loss = sum(loss) / float(len(loss))
        loss.backward(retain_graph=(update_type == "actor"))
        optimizer.step()

        # zero_grad(network)
        # #optimizer.zero_grad()
        # loss.backward(retain_graph=(update_type == "actor"))
        # torch.nn.utils.clip_grad_norm_(get_parameters(network), self.args.max_grad_clip)

        # # Apply projection conflicting gradient
        # process_dict[str(self.rank) + key] = \
        #     np.copy(to_vector([param.grad for param in get_parameters(network)]).detach().numpy())
        # projected_grad = pc_grad(process_dict, self.rank, self.args, key)
        # to_parameters(torch.from_numpy(projected_grad), [param._grad for param in get_parameters(network)])
        #
        # # Update networks
        # ensure_shared_grads(network, shared_network)
        # shared_optimizer.step()
        #
        # # For logging
        # if tb_key is not None:
        #     self.tb_writer.add_scalars(tb_key, {"agent" + str(self.i_agent): loss.data.numpy()}, iteration)

    def observe_oppo_action(self, state, oppo_action, iteration, no_log=True):
        '''
        :param state: np.ndarray, [1, n_state]
        :param oppo_action: np.ndarray or int [1]
        :return: None
        '''
        assert type(state) is np.ndarray, "observe_oppo_action input type error"
        #assert (type(oppo_action) is np.ndarray) or (type(oppo_action) is int), "observe_oppo_action input type error"
        #assert state.shape == (1, self.conf["n_state"]), "observe_oppo_action input shape error"
        with torch.no_grad():
            if type(oppo_action) is not np.ndarray:
                oppo_action = np.array([oppo_action])
            oppo_action = oppo_action.reshape(1, 1)
            assert oppo_action.shape == (1, 1), "observe_oppo_action input shape error"
            state = state.reshape(1, -1)
            state = torch.Tensor(state).to(device=self.device)
            oppo_action = torch.LongTensor(oppo_action).to(device=self.device)
            # store experience
            self.om_buffer.store_memory(state, oppo_action)
            # learn model phi_0
            data = self.om_buffer.get_batch(self.conf["opponent_model_batch_size"])
        phi, loss = self.oppo_model.learn(data, self.oppo_model.get_parameter(),
                                          lr=self.conf["opponent_model_learning_rate"],
                                          l_times=self.conf["opponent_model_learning_times"])
        self.oppo_model.set_parameter(phi)
        if not no_log:
            self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, Loss_oppo=loss)
        pass

    def save_model(self, iteration):
        import os
        filepath = os.path.join(self.logger.model_dir, self.name + "_iter" + str(iteration) + ".ckp")
        obj = {
            'v_net_state_dict': self.v_net.state_dict(),
            'a_net_state_dict': self.a_net.state_dict(),
            'v_optimizer_state_dict': self.v_optimizer.state_dict(),
            'a_optimizer_state_dict': self.a_optimizer.state_dict(),
            'args': self.args,
            'conf': self.conf,
            'name': self.name,
            'agent_idx': self.agent_idx,
            'actor_rnn': self.actor_rnn,
        }
        torch.save(obj, filepath)

        self.logger.log("model saved in {}".format(filepath))

    @staticmethod
    def load_model(filepath, args, logger, device, **kwargs):
        assert "env_model" in kwargs.keys(), "must input env_model"
        checkpoint = torch.load(filepath)
        #args = checkpoint["args"]
        conf = checkpoint["conf"]
        name = checkpoint["name"].replace("MBAM_", "")
        agent_idx = checkpoint["agent_idx"]
        actor_rnn = checkpoint["actor_rnn"]
        mbam = Meta_mapg(args, conf, name, logger, agent_idx, actor_rnn, device)

        mbam.v_net.load_state_dict(checkpoint['v_net_state_dict'])
        mbam.a_net.load_state_dict(checkpoint['a_net_state_dict'])
        mbam.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
        mbam.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])

        if device:
            mbam.v_net = mbam.v_net.to(device)
            mbam.a_net = mbam.a_net.to(device)
        if logger is not None:
            logger.log("model successful load, {}".format(filepath))
        return mbam

class Meta_mapg_Buffer(object):
    def __init__(self, args, conf, name, actor_rnn, device=None):
        self.args = args
        self.conf = conf
        self.name = name
        self.actor_rnn = actor_rnn
        self.device = device
        self.gamma = conf["gamma"]
        self.lam = conf["lambda"]

        self.state = torch.zeros((conf["n_trj"], conf["trj_length"], conf["n_state"]), dtype=torch.float32)
        self.action = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.int)
        self.reward = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32)
        self.reward_to_go = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32)
        self.advantage = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32)
        self.logp_a = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32, device=self.device, requires_grad=True)
        #self.logp_a = []
        self.value = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32)
        if self.actor_rnn:
            self.hidden_state = torch.zeros((conf["n_trj"], conf["trj_length"], self.conf["a_hidden_layers"][0]), dtype=torch.float32)
        self.oppo_logp_a = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32)
        self.mask = torch.zeros((conf["n_trj"], conf["trj_length"], 1), dtype=torch.float32)

        self.next_idx, self.max_size = 0, conf["n_trj"]

    def store_memory(self, episode_memory, last_val=0):
        '''
        :param data : state, action, reward ,logp_a, value, hidden_state, oppo_hidden_prob
                    state: np.ndarray shape is (n_batch, n_state) float
                    action: np.ndarray shape is (n_batch, 1) int
                    reward: np.ndarray shape is (n_batch, 1) float
                    logp_a: np.ndarray shape is (n_batch, 1) float
                    value: np.ndarray shape is (n_batch, 1) float
                    hidden_state: np.ndarray shape is (n_batch, 1) float
                    oppo_logp_a: np.ndarray shape is (n_batch, 1) float
        :param last_val The "last_val" argument should be 0 if the trajectory ended
                        because the agent reached a terminal state (died), and otherwise
                        should be V(s_T), the value function estimated for the last state.
        :return: None
        '''
        def expend_to_max_length(x, max_length):
            assert len(x.shape) == 2, ""
            temp = torch.zeros((max_length - x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, temp], dim=0)
            return x

        data = episode_memory.get_data(is_meta_mapg=True)
        n_batch = data["state"].shape[0]
        for k in data.keys():
            assert data[k].shape[0] == n_batch, "input size error"

        # cal GAE-Lambda advantage and rewards-to-go
        reward = data["reward"]
        if type(reward) is torch.Tensor: reward = dcn(reward)
        value = data["value"]
        if type(value) is torch.Tensor: value = dcn(value)

        reward_l = np.append(reward, last_val)
        value_l = np.append(value, last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = reward_l[:-1] + self.gamma * value_l[1:] - value_l[:-1]
        advantage = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        reward_to_go = discount_cumsum(reward_l, self.gamma)[:-1]

        #store
        assert self.next_idx < self.max_size, "Buffer {} Full!!!".format(self.name)
        if self.actor_rnn:
            hidden_state = data["hidden_state"]
            if type(hidden_state) is np.ndarray: hidden_state = torch.Tensor(hidden_state)
            hidden_state = expend_to_max_length(hidden_state, self.args.eps_max_step)
            self.hidden_state[self.next_idx] = hidden_state

        action = data["action"]
        if type(action) is np.ndarray: action = torch.LongTensor(action)
        action = expend_to_max_length(action, self.args.eps_max_step)
        self.action[self.next_idx] = action

        state = data["state"]
        if type(state) is np.ndarray: state = torch.Tensor(state)
        state = expend_to_max_length(state, self.args.eps_max_step)
        self.state[self.next_idx] = state

        logp_a = data["logp_a"]
        if type(logp_a) is np.ndarray: logp_a = torch.Tensor(logp_a)
        logp_a = expend_to_max_length(logp_a, self.args.eps_max_step)
        #self.logp_a.append(logp_a)
        self.logp_a[self.next_idx] = logp_a

        advantage = advantage.copy()
        advantage = torch.Tensor(advantage).view(-1, 1)
        advantage = expend_to_max_length(advantage, self.args.eps_max_step)
        self.advantage[self.next_idx] = advantage

        reward_to_go = reward_to_go.copy()
        reward_to_go = torch.Tensor(reward_to_go).view(-1, 1)
        reward_to_go = expend_to_max_length(reward_to_go, self.args.eps_max_step)
        self.reward_to_go[self.next_idx] = reward_to_go

        oppo_logp_a = data["oppo_logp_a"]
        if type(oppo_logp_a) is np.ndarray: oppo_logp_a = torch.Tensor(oppo_logp_a)
        oppo_logp_a = expend_to_max_length(oppo_logp_a, self.args.eps_max_step)
        self.oppo_logp_a[self.next_idx] = oppo_logp_a

        mask = torch.full((n_batch, 1), fill_value=1.0)
        mask = expend_to_max_length(mask, self.args.eps_max_step)
        self.mask[self.next_idx] = mask

        self.next_idx = self.next_idx + 1
        pass

    def store_multi_memory(self, data, last_val=0):
        '''
        :param data : list, for episode_memory
        :param last_val
        :return: None
        '''
        if type(data) == list:
            for i, d in enumerate(data):
                if (type(last_val) is not list) and (type(last_val) is not np.ndarray):
                    self.store_memory(d, last_val)
                else:
                    self.store_memory(d, last_val[i])
        else:
            self.store_memory(data, last_val)
    def clear_memory(self):
        self.next_idx = 0

    def get_batch(self, batch_size=0):
        with torch.no_grad():
            state = self.state[:self.next_idx].to(device=self.device)
            action = self.action[:self.next_idx].to(device=self.device)
            reward_to_go = self.reward_to_go[:self.next_idx].to(device=self.device)
            advantage = self.advantage[:self.next_idx].to(device=self.device)
            adv_mean, adv_std = torch.mean(advantage, dim=0), torch.std(advantage, dim=0)
            adv_std[adv_std == 0.0] = 1.0
            advantage = (advantage - adv_mean) / adv_std
            adv_mean = adv_mean.cpu()
            adv_std = adv_std.cpu()
            logp_a = self.logp_a
            oppo_logp_a = self.oppo_logp_a[:self.next_idx].to(device=self.device)
            mask = self.mask[:self.next_idx].to(device=self.device)
            data = dict({"state": state,
                         "action": action,
                         "reward_to_go": reward_to_go,
                         "advantage": advantage,
                         "logp_a": logp_a,
                         "oppo_logp_a": oppo_logp_a,
                         "mask": mask,})
            if self.actor_rnn:
                hidden_state = self.hidden_state[:self.next_idx].to(device=self.device)
                data["hidden_state"] = hidden_state

            self.next_idx = 0
            return data



if __name__ == "__main__":
    import time
    import argparse
    import numpy as np
    import os
    from DRL_MARL_homework.MBAM.utils.get_exp_data_path import get_exp_data_path
    from DRL_MARL_homework.MBAM.utils.rl_utils import collect_trajectory
    from DRL_MARL_homework.MBAM.env_wapper.football_penalty_kick.football_1_vs_1_penalty_kick import make_env as football_env
    from DRL_MARL_homework.MBAM.utils.Logger import Logger
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--exp_name", type=str, default="football_ppo_vs_mbam", help="football_ppo_vs_mbam\n " +
                                                                                     "predator_ppo_vs_mbam\n")
    parser.add_argument("--env", type=str, default="football", help="football\n" +
                                                                    "predator\n")
    parser.add_argument("--prefix", type=str, default="test", help="train or test")

    parser.add_argument("--seed", type=int, default=-1, help="-1 means random seed")
    parser.add_argument("--ranks", type=int, default=1, help="for prefix is train")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--dir", type=str, default="", help="")

    parser.add_argument("--eps_max_step", type=int, default=30, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=5, help="")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="")
    parser.add_argument("--max_epoch", type=int, default=10000, help="train epoch")

    parser.add_argument("--num_om_layers", type=int, default=3, help="opponent layers")
    parser.add_argument("--actor_rnn", type=bool, default=True, help="True or False")
    parser.add_argument("--true_prob", type=bool, default=True, help="True or False")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--meta_oppo_modeling", type=bool, default=True, help="True or False")
    args = parser.parse_args()
    conf = {
        "conf_id": "goalkeeper_conf",
        # env setting
        "n_state": 24,
        "n_action": 11,
        "n_opponent_action": 11,
        "action_dim": 1,
        "type_action": "discrete",  # "discrete", "continuous"
        "action_bounding": 0,  # [()]
        "action_scaling": [1, 1],
        "action_offset": [0, 0],

        # opponent model setting
        "opponent_model_hidden_layers": [32, 16],
        "opponent_model_memory_size": 1000,
        "opponent_model_learning_rate": 0.001,
        "opponent_model_batch_size": 32,
        "opponent_model_learning_times": 1,

        "opponent_optimal_model_learning_rate": 0.001,
        "opponent_optimal_model_learning_times": 8,

        "imagine_model_learning_rate": 0.001,
        "imagine_model_learning_times": 2,
        # times is 0 means training until optimal action is max prob , but unfinished
        "roll_out_length": 5,
        "short_term_decay": 0.9,  # error discount factor
        "short_term_horizon": 10,
        "mix_factor": 0.01,  # adjust mix cuvre

        # ppo setting
        "v_hidden_layers": [32, 16],
        "a_hidden_layers": [32, 16],
        "v_learning_rate": 0.001,
        "a_learning_rate": 0.001,
        "gamma": 0.99,  # value discount factor
        "lambda": 0.99,  # general advantage estimator
        "epsilon": 0.115,  # ppo clip param
        "entcoeff": 0.0015,
        "a_update_times": 10,
        "v_update_times": 10,
        # ppo buffer setting
        "buffer_memory_size": 3000, #update_episode * max_episode_step = 20 * 30
        "update_episode": 50,

        # meta-mapg
        "n_trj": 5,
        "trj_length": 30,
        "chain_horizon": 3,
        "actor_lr_inner": 0.01,
        "actor_lr_outer": 0.001,
    }
    BATCH_SIZE = 1

    env = football_env()
    logger = Logger("./temp/", "QWERQWERQWER", 0)
    meta_mapg = Meta_mapg(args=args, conf=conf, name="AAA", logger=logger, agent_idx=1, actor_rnn=True, device=args.device)
    meta_mapg_buffer = Meta_mapg_Buffer(args, conf, meta_mapg.name, True, "cuda")
    ppo = PPO(args, conf, name="AAA", logger=logger, actor_rnn=args.actor_rnn, device=args.device)
    ppo_buffer = PPO_Buffer(args=args, conf=ppo.conf, name=ppo.name, actor_rnn=args.actor_rnn, device=args.device)

    shooters_path = get_exp_data_path() + "/Football_Penalty_Kick/Shooter"
    shooter_model_file_list = []
    for root, dirs, files in os.walk(shooters_path):
        for f in files:
            shooter_model_file_list.append(os.path.join(root, f))

    iteration = 0
    global_step = 0
    while True:
        iteration += 1
        # Sync thread-specific meta-agent with shared meta-agent
        a_phi = meta_mapg.get_a_parameter()
        # Set opponent's persona
        ppo.load_model(shooter_model_file_list[iteration % len(shooter_model_file_list)], args, None, args.device)
        # Accumulate actor and value losses for outer-loop optimization
        # through processing until the end of Markov chain
        actor_losses, value_losses = [], []

        agents = [ppo, meta_mapg]

        datum = []
        phis = [meta_mapg.get_a_parameter()]
        for i_joint in range(meta_mapg.conf["chain_horizon"] + 1):

            # Collect trajectory
            memories, scores, global_step = collect_trajectory(agents, env, args, global_step, is_prophetic=False)
            # memories [2, num_trj, trj_length]
            # Perform inner-loop update
            # phis = []
            # for agent in zip(agents):
            #     phi = agent.inner_update(actor, memory, i_joint, is_train=True)
            #     phis.append(phi)
            ppo_buffer.store_multi_memory(memories[0], last_val=0)
            meta_mapg_buffer.store_multi_memory(memories[1], last_val=0)

            ppo.learn(ppo_buffer.get_batch(), iteration, no_log=True)
            data = meta_mapg_buffer.get_batch()
            datum.append(data)
            new_phi = meta_mapg.inner_update(data, phis[-1], i_joint, is_train=True, iteration=iteration, no_log=True)

            # Compute outer-loop loss
            actor_loss, value_loss = meta_mapg.compute_outer_loss(datum=datum, iteration=iteration, no_log=True)
            actor_losses.append(actor_loss)
            value_losses.append(value_loss)

            # For next round
            if new_phi is not None:
                meta_mapg.set_parameter(new_phi.values())
                phis.append(new_phi)

        # Perform outer update
        meta_mapg.outer_update(actor_losses, iteration, update_type="actor")
        meta_mapg.outer_update(actor_losses, iteration, update_type="dynamic_lr")
        meta_mapg.outer_update(value_losses, iteration, update_type="value")