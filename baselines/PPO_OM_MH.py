import sys
sys.path.append("/home/lenovo/文档/CodeWorkspace/RL")
from baselines.Base_ActorCritic import Base_ActorCritic
from baselines.Base_ActorCritic_MH import Base_ActorCritic_MH
from utils.datatype_transform import dcn
from utils.rl_utils import discount_cumsum
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from memory_profiler import profile

#源代码class PPO_OM_MH(Base_ActorCritic):
class PPO_OM_MH(Base_ActorCritic_MH):
    def __init__(self, args, conf, name, logger, actor_rnn, device=None):
        super(PPO_OM_MH, self).__init__(a_n_state=conf["n_state"],
                                  v_n_state=conf["n_state"],
                                  n_action=conf["n_action"],
                                  a_hidden_layers=conf["a_hidden_layers"],
                                  v_hidden_layers=conf["v_hidden_layers"],
                                  actor_rnn=actor_rnn,
                                  args=args,
                                  conf=conf,
                                  name="PPO_" + name,
                                  logger=logger)

        self.device = device
        if device is not None:
            self.change_device(device)

    def init_hidden_state(self, n_batch):
        if self.actor_rnn:
            hidden_state = torch.zeros((n_batch, self.conf["a_hidden_layers"][0]), device=self.device)
            return hidden_state
        else:
            return None

    def choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
        assert (type(state) is np.ndarray) or (type(state) is torch.Tensor), "choose_action input type error"
        #with torch.no_grad():
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
            #oppo_hidden_prob = oppo_hidden_prob.view((-1, sum(self.conf["n_opponent_action"]) if self.args.true_prob else self.conf["opponent_model_hidden_layers"][-1]))
            oppo_hidden_prob = oppo_hidden_prob.view((-1, self.conf["opponent_model_hidden_layers"][-1]))
            
            if self.args.prophetic_onehot:
                oppo_hidden_prob = torch.eye(self.conf["n_opponent_action"], device=self.device)[torch.argmax(oppo_hidden_prob, dim=1)]
            #print("a_state" ,  a_state)
            #print("hidden_prob" ,  oppo_hidden_prob)
            a_state = torch.cat([a_state, oppo_hidden_prob], dim=1)
            #print("pinjie" ,  a_state)
        if hidden_state is not None:
            if type(hidden_state) is np.ndarray:
                hidden_state = torch.Tensor(hidden_state).to(device=self.device)
            hidden_state = hidden_state.view((-1, self.conf["a_hidden_layers"][0]))
        value = self.v_net(v_state)
        if self.actor_rnn:
            action_prob, hidden_prob, hidden_state = self.a_net(a_state, hidden_state)
        else:
            action_prob, hidden_prob = self.a_net(a_state)
        #print("action prob" , action_prob)
        #print("hidden_prob222",hidden_prob)
        if greedy:
            pi = Categorical(action_prob)
            _, action = torch.max(action_prob, dim=1)
            logp_a = pi.log_prob(action)
            entropy = pi.entropy()
        else:
            pi = [torch.distributions.Categorical(prob) for prob in action_prob]
            action = [p.sample() for p in pi]
            logp_a = [p.log_prob(a) for p, a in zip(pi, action)]
            entropy = [p.entropy() for p in pi]

        return ([dcn(a).astype(np.int32) for a in action],
                [dcn(l) for l in logp_a],
                [dcn(ent) for ent in entropy],
                dcn(value),
                [dcn(acp) for acp in action_prob],
                [dcn(hip) for hip in hidden_prob],
                dcn(hidden_state) if self.actor_rnn else None)        
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
        
        assert (type(state) is np.ndarray) or (type(state) is torch.Tensor), "choose_action input type error"
        #with torch.no_grad():
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
            oppo_hidden_prob = oppo_hidden_prob.view((-1, sum(self.conf["n_opponent_action"]) if self.args.true_prob else self.conf["opponent_model_hidden_layers"][-1]))
            if self.args.prophetic_onehot:
                oppo_hidden_prob = torch.eye(self.conf["n_opponent_action"], device=self.device)[torch.argmax(oppo_hidden_prob, dim=1)]
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
        action_prob = torch.tensor(action_prob)
        print("action prob" , action_prob)
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
                dcn(logp_a),
                dcn(entropy),
                dcn(value),
                dcn(action_prob),
                dcn(hidden_prob),
                dcn(hidden_state) if self.actor_rnn else None)
        '''#源代码
    def single_choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
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
        raise NotImplementedError
        assert (type(state) is np.ndarray) or (type(state) is torch.Tensor), "choose_action input type error"
        #with torch.no_grad():
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
            if self.args.prophetic_onehot:
                oppo_hidden_prob = torch.eye(self.conf["n_opponent_action"], device=self.device)[
                    torch.argmax(oppo_hidden_prob, dim=1)]
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
        # print("action prob" , action_prob)
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
                entropy,
                value,
                action_prob,
                hidden_prob,
                hidden_state if self.actor_rnn else None)

    def change_device(self, device):
        self.device = device
        self.a_net = self.a_net.to(device)
        self.v_net = self.v_net.to(device)

    def learn(self, data, iteration, no_log=True):
        '''state, action, reward_to_go, advantage, logp_a, hidden_state, oppo_action_prob
        :param state: torch.Tensor shape is (n_batch, n_state) float
               action: torch.Tensor shape is (n_batch, 1) int
               reward_to_go: torch.Tensor shape is (n_batch, 1) float
               logp_a: torch.Tensor shape is (n_batch, 1) float
               advantage: torch.Tensor shape is (n_batch, 1) float
               hidden_state: torch.Tensor shape is (n_batch, 1) float
               oppo_hidden_prob: torch.Tensor shape is (n_batch, 1) float
        :param iteration:
        '''

        for param_group in self.a_optimizer.param_groups:
            param_group['lr'] = self.conf["a_learning_rate"]
        for param_group in self.v_optimizer.param_groups:
            param_group['lr'] = self.conf["v_learning_rate"]

        # data type is np.ndarray
        state = data["state"]
        a_state = state.clone()
        v_state = state
        if "MBAM" in self.name:
            oppo_hidden_prob = data["oppo_hidden_prob"]
            oppo_hidden_prob = torch.cat(oppo_hidden_prob, dim=1)
            a_state = torch.cat([a_state, oppo_hidden_prob], dim=1)
        if self.actor_rnn:
            hidden_state = data["hidden_state"]
        action = data["action"]
        #for i, lp in enumerate(action):
        #                 print(f"Shapeeee of action[{i}]:", lp.shape)
        reward_to_go = data["reward_to_go"]
        advantage = data["advantage"]
        logp_a = data["logp_a"]

        if self.device:
            a_state = a_state.to(self.device)
            v_state = v_state.to(self.device)
            #源代码action = action.to(self.device)
            
            action = [a.to(self.device) for a in action]
            
            reward_to_go = reward_to_go.to(self.device)
            advantage = advantage.to(self.device)
            #源代码logp_a = logp_a.to(self.device)
            
            logp_a = [lpa.to(self.device) for lpa in logp_a]
            
            if self.actor_rnn:
                hidden_state = hidden_state.to(self.device)

        def compute_loss_a(state, action, advantage, logp_old):
            # Policy loss
            if self.conf["type_action"] == "discrete":
                if self.actor_rnn:
                    prob, _, _ = self.a_net(state, hidden_state)
                else:
                    prob, _ = self.a_net(state)
                #for i, lp in enumerate(prob):
                #    print(f"Shape of prob[{i}]:", lp.shape)
                #源代码pi = Categorical(prob)
                #源代码logp = pi.log_prob(action.squeeze())
                pi = [Categorical(p) for p in prob]
                #logp = [p.log_prob(a.squeeze()) for p, a in zip(pi, action)]
                '''
                for i, lp in enumerate(action):
                         print(f"Shape前 of action[{i}]:", lp.shape)

# 将列表中的张量堆叠成一个形状为 [5000, 6, 1] 的三维张量
                stacked_tensor = torch.stack(action)

# 转置堆叠后的张量，以获得形状为 [6, 5000, 1] 的张量
                transposed_tensor = stacked_tensor.permute(1, 0, 2)

# 将形状为 [6, 5000, 1] 的张量分解为 6 个形状为 [5000, 1] 的张量
                action = [transposed_tensor[i] for i in range(6)]
                print("action",type(action))
                print("action[0]",type(action[0]))
                for i, lp in enumerate(action):
                         print(f"Shape后 of action[{i}]:", lp.shape)
                sys.exit()
                '''
                logp = [p.log_prob(a.squeeze()) for p, a in zip(pi, action)]#[6,6,5000]
                #print("yunxingok")


            else:  # "continuous"
                raise NotImplementedError
                #mu, sigma = self.a_net(state)
                #pi = torch.distributions.Normal(mu, sigma)
                #logp = pi.log_prob(action.squeeze()).sum(axis=-1)
            #源代码logp_old = logp_old.squeeze()
            logp_old = [l_old.squeeze() for l_old in logp_old]
            #源代码assert logp.shape == logp_old.shape, "compute_loss_a error! logp.shape != logp_old.shape"
            #源代码ratio = torch.exp(logp - logp_old.squeeze())
            ratio = [torch.exp(l - l_old.squeeze()) for l, l_old in zip(logp, logp_old)]
            advantage = advantage.squeeze()
            #源代码assert ratio.shape == advantage.shape, "compute_loss_a error! ratio.shape != advantage.shape"
            #源代码clip_advantage = torch.clamp(ratio, 1 - self.conf["epsilon"], 1 + self.conf["epsilon"]) * advantage
            #源代码ent = pi.entropy().mean()
            #源代码loss_a = -(torch.min(ratio * advantage, clip_advantage)).mean() - self.conf["entcoeff"] * ent
            clip_advantage = [torch.clamp(r, 1 - self.conf["epsilon"], 1 + self.conf["epsilon"]) * advantage for r in ratio]
            ent = [p.entropy().mean() for p in pi]
            loss_a = torch.stack([-(torch.min(r * advantage, c_adv)).mean() - self.conf["entcoeff"] * en for r, c_adv, en in zip(ratio, clip_advantage, ent)]).mean()

            # extra info
            with torch.no_grad():
                '''源代码approx_kl = (logp_old - logp).mean().item()
                approx_kl = sum([(l_old - l).mean().item() for l_old, l in zip(logp_old, logp)])/len(logp)
                ent_info = ent.item()
                clipped = ratio.gt(1 + self.conf["epsilon"]) | ratio.lt(1 - self.conf["epsilon"])
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                a_net_info = dict(kl=approx_kl, ent=ent_info, cf=clipfrac)
                '''
                approx_kl = sum([(l_old - l).mean().item() for l_old, l in zip(logp_old, logp)])/len(logp)
                ent_info = sum([e.item() for e in ent])/len(ent)
                clipped = torch.stack([ratio[i].gt(1 + self.conf["epsilon"]) | ratio[i].lt(1 - self.conf["epsilon"]) for i in range(len(ratio))])
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                a_net_info = dict(kl=approx_kl, ent=ent_info, cf=clipfrac)
            return loss_a, a_net_info
        def compute_loss_v(state, reward_to_go):
            return ((self.v_net(state) - reward_to_go) ** 2).mean()

        loss_a_old, a_info_old = compute_loss_a(a_state, action, advantage, logp_a)
        loss_a_old = loss_a_old.detach().item()
        loss_v_old = compute_loss_v(v_state, reward_to_go).detach().item()

        for _ in range(self.conf["a_update_times"]):
            self.a_optimizer.zero_grad()
            loss_a, _ = compute_loss_a(a_state, action, advantage, logp_a)
            loss_a.backward()
            self.a_optimizer.step()
        for _ in range(self.conf["v_update_times"]):
            self.v_optimizer.zero_grad()
            loss_v = compute_loss_v(v_state, reward_to_go)
            loss_v.backward()
            self.v_optimizer.step()
        if not no_log:
            self.logger.log_performance(tag=self.name + "/epochs", iteration=iteration,
                                        Loss_a=loss_a_old, Loss_v=loss_v_old,
                                        KL=a_info_old['kl'], Entropy=a_info_old['ent'],
                                        # ClipFrac=a_info_old['cf'],
                                        # DeltaLossPi=(loss_a.item() - loss_a_old),
                                        # DeltaLossV=(loss_v.item() - loss_v_old)
                                        )
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
            'actor_rnn': self.actor_rnn,
        }
        torch.save(obj, filepath, _use_new_zipfile_serialization=False)
        self.logger.log("model saved in {}".format(filepath))

    @staticmethod
    def load_model(filepath, args, logger, device, **kwargs):
        checkpoint = torch.load(filepath, map_location='cpu')
        #args = checkpoint["args"]
        conf = checkpoint["conf"]
        name = checkpoint["name"].replace("PPO_", "")
        actor_rnn = checkpoint["actor_rnn"]
        ppo = PPO_OM_MH(args, conf, name, logger, actor_rnn, device)

        ppo.v_net.load_state_dict(checkpoint['v_net_state_dict'])
        ppo.a_net.load_state_dict(checkpoint['a_net_state_dict'])
        ppo.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
        ppo.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])

        if device:
            ppo.v_net = ppo.v_net.to(device)
            ppo.a_net = ppo.a_net.to(device)
        if logger is not None:
            logger.log("model successful load, {}".format(filepath))
        return ppo
class PPO_OM_MH_Buffer(object):
    def __init__(self, args, conf, name, actor_rnn, device=None):
        self.args = args
        self.conf = conf
        self.name = name
        self.actor_rnn = actor_rnn
        self.device = device
        self.gamma = conf["gamma"]
        self.lam = conf["lambda"]
        self.oppo_output_dim = len(conf["n_opponent_action"])
        self.output_dim = len(conf["n_action"])

        self.state = torch.zeros((conf["buffer_memory_size"], conf["n_state"]), dtype=torch.float32)
        #源代码self.action = torch.zeros((conf["buffer_memory_size"], 1), dtype=int)
        self.action = [torch.zeros((conf["buffer_memory_size"], 1), dtype=int) for _ in range(self.output_dim)]
        self.reward = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.reward_to_go = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.advantage = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        #源代码self.logp_a = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.logp_a = [torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32) for _ in range(self.output_dim)]
        self.value = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        if self.actor_rnn:
            self.hidden_state = torch.zeros((conf["buffer_memory_size"], self.conf["a_hidden_layers"][0]), dtype=torch.float32)
        if "MBAM" in self.name:
            self.oppo_hidden_prob = [torch.zeros((conf["buffer_memory_size"], self.conf["n_opponent_action"][i]) if self.args.true_prob else (conf["buffer_memory_size"], self.conf["opponent_model_hidden_layers"][-1]), dtype=torch.float32) for i in range(self.oppo_output_dim)]

        self.next_idx, self.max_size = 0, conf["buffer_memory_size"]

    def store_memory(self, episode_memory, last_val=0):
        '''
        :param data : state, action, reward ,logp_a, value, hidden_state, oppo_hidden_prob
                    state: np.ndarray shape is (n_batch, n_state) float
                    action: np.ndarray shape is (n_batch, 1) int
                    reward: np.ndarray shape is (n_batch, 1) float
                    logp_a: np.ndarray shape is (n_batch, 1) float
                    value: np.ndarray shape is (n_batch, 1) float
                    hidden_state: np.ndarray shape is (n_batch, 1) float
                    oppo_hidden_prob: np.ndarray shape is (n_batch, 1) float
        :param last_val The "last_val" argument should be 0 if the trajectory ended
                        because the agent reached a terminal state (died), and otherwise
                        should be V(s_T), the value function estimated for the last state.
        :return: None
        '''
        data = episode_memory.get_data()
        n_batch = data["state"].shape[0]

        #for k in data.keys():
        #    if k is not 'oppo_hidden_prob':
        #        assert data[k].shape[0] == n_batch, "input size error"

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
        path_slice = slice(self.next_idx, self.next_idx + n_batch)
        assert self.next_idx + n_batch <= self.max_size, "Buffer {} Full!!!".format(self.name)
        if self.actor_rnn:
            hidden_state = data["hidden_state"]
            if type(hidden_state) is np.ndarray: hidden_state = torch.Tensor(hidden_state)
            self.hidden_state[path_slice] = hidden_state

        if "MBAM" in self.name:
            oppo_hidden_prob = data["oppo_hidden_prob"]
            #将(n_batch,n_oppo_num，action_prob)转换为（n_oppo_num，n_batch，action_prob）
            if type(oppo_hidden_prob[0]) is np.ndarray: oppo_hidden_prob = torch.Tensor(oppo_hidden_prob)
            oppo_hidden_prob = oppo_hidden_prob.permute(1, 0, 2)
            for i in range(self.oppo_output_dim):#智能体个数
                '''
                print("path_slice:", path_slice)#(0, 100, None)
                print(self.oppo_hidden_prob[i].shape)#([6000, 5])
                #print(self.oppo_hidden_prob.shape)#list
                print(self.oppo_hidden_prob[i][path_slice].shape)#([100, 5])
                print(oppo_hidden_prob.shape)#([100, 6, 5])
                print(oppo_hidden_prob[i].shape)#([6, 5])
                '''
                self.oppo_hidden_prob[i][path_slice] = oppo_hidden_prob[i]

        action = data["action"]
        #源代码if type(action) is np.ndarray: action = torch.LongTensor(action)        
        #源代码self.action[path_slice] = action
        if type(action[0]) is np.ndarray:
            action = [torch.LongTensor(a) for a in action]
        '''
        for i, lp in enumerate(action):
                 print(f"Shape of actionnnn[{i}]:", lp.shape)
        print(action[0][path_slice].shape)
        sys.exit()维数有误？
        '''
        for i in range(self.output_dim):
            self.action[i][path_slice] = action[i]


        state = data["state"]
        if type(state) is np.ndarray: state = torch.Tensor(state)
        self.state[path_slice] = state

        logp_a = data["logp_a"]
        #源代码if type(logp_a) is np.ndarray: logp_a = torch.Tensor(logp_a)
        #源代码self.logp_a[path_slice] = logp_a
        if type(logp_a[0]) is np.ndarray:
            logp_a = [torch.Tensor(l) for l in logp_a]
        for i in range(self.output_dim):
            self.logp_a[i][path_slice] = logp_a[i]


        advantage = advantage.copy()
        advantage = torch.Tensor(advantage).view(-1, 1)
        self.advantage[path_slice] = advantage

        reward_to_go = reward_to_go.copy()
        reward_to_go = torch.Tensor(reward_to_go).view(-1, 1)
        self.reward_to_go[path_slice] = reward_to_go

        self.next_idx = self.next_idx + n_batch
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
        state = self.state[:self.next_idx].to(device=self.device)
        #源代码action = self.action[:self.next_idx].to(device=self.device)
        action = [a[:self.next_idx].to(device=self.device) for a in self.action]#500*6*1
        #for i, lp in enumerate(self.action):
         #              print(f"Sha2peeee of self.action[{i}]:", lp.shape)
        #sys.exit()
        reward_to_go = self.reward_to_go[:self.next_idx].to(device=self.device)
        advantage = self.advantage[:self.next_idx].to(device=self.device)
        adv_mean, adv_std = torch.mean(advantage), torch.std(advantage)
        if adv_std == 0.0:
            advantage = (advantage - adv_mean)
            adv_mean = adv_mean.detach().cpu()
            adv_std = adv_std.detach().cpu()
        else:
            advantage = (advantage - adv_mean) / adv_std
            adv_mean = adv_mean.detach().cpu()
            adv_std = adv_std.detach().cpu()
        #源代码logp_a = self.logp_a[:self.next_idx].to(device=self.device)

        logp_a = [l[:self.next_idx].to(device=self.device) for l in self.logp_a]

        data = dict({"state": state,
                     "action": action,
                     "reward_to_go": reward_to_go,
                     "advantage": advantage,
                     "logp_a": logp_a})
        if self.actor_rnn:
            hidden_state = self.hidden_state[:self.next_idx].to(device=self.device)
            data["hidden_state"] = hidden_state
        if "MBAM" in self.name:
            oppo_hidden_prob = [self.oppo_hidden_prob[i][:self.next_idx].to(device=self.device) for i in range(self.oppo_output_dim)]
            data["oppo_hidden_prob"] = oppo_hidden_prob
        self.next_idx = 0
        return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="test")
    from env_wapper.simple_predator import simple_predator
    args = parser.parse_args()
    conf = {
        "conf_id": "shooter_conf",
        # env setting
        "n_state": 24,
        "n_action": 11,
        "n_opponent_action": 11,
        "action_dim": 1,
        "type_action": "discrete",  # "discrete", "continuous"
        "action_bounding": 0,  # [()]
        "action_scaling": [1, 1],
        "action_offset": [0, 0],

        # shooter ppo setting
        "v_hidden_layers": [32, 16],
        "a_hidden_layers": [32, 16],
        "v_learning_rate": 0.001,
        "a_learning_rate": 0.001,
        "gamma": 0.99,  # value discount factor
        "lambda": 0.99,  # general advantage estimator
        "epsilon": 0.115,  # ppo clip param
        "entcoeff": 0.0015,
        "a_update_times": 3,
        "v_update_times": 3,
        "buffer_memory_size": 3000,  # update_episode * max_episode_step = 20 * 30
        "update_episode": 10,
    }
    torch.cuda.is_available()
    ppo = PPO(args=args, conf=conf, name="AAA", logger=None, actor_rnn=True, device="cuda")
    hidden_state = ppo.init_hidden_state(n_batch=5)
    for i in range(10):
        state = np.random.random((5, 24))
        print("ppo choose action:{}".format(i))
        action, logp_a, entropy, value, action_prob, hidden_prob, hidden_state = ppo.choose_action(state, greedy=False, hidden_state=hidden_state)
    print("learn test")

    ppobf = PPO_Buffer(args, conf, "123123", True, "cuda")
    for i in range(100000):
        BATCH_SIZE = np.random.randint(300)
        state = np.random.random((BATCH_SIZE, 24))
        action = np.random.randint(0, 10, (BATCH_SIZE, 1))
        reward = np.random.random((BATCH_SIZE, 1))
        value = np.random.random((BATCH_SIZE, 1))
        logp_a = np.random.random((BATCH_SIZE, 1))
        hidden_state = np.random.random((BATCH_SIZE, 32))
        oppo_hidden_prob = np.random.random((BATCH_SIZE, 16))
        data = dict({"state": state,
                     "action": action,
                     "reward": reward,
                     "value": value,
                     "logp_a": logp_a,
                     "hidden_state": hidden_state,
                     "oppo_hidden_prob": oppo_hidden_prob,})
        ppobf.store_memory(data, last_val=0)
        print(ppobf.next_idx)
        if ppobf.next_idx > 500:
            data = ppobf.get_batch()
            ppo.learn(data, 100, no_log=True)
    print("end")
    pass