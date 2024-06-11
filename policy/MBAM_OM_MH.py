import sys
sys.path.append("/home/lenovo/文档/CodeWorkspace/RL")
from baselines.PPO_OM_MH import PPO_OM_MH
from policy.Opponent_Model_MH import Opponent_Model_MH, OM_MH_Buffer
from utils.torch_tool import soft_update
import torch
import numpy as np
from memory_profiler import profile

class MBAM_OM_MH(PPO_OM_MH):
    def __init__(self, args, conf, name, logger, agent_idx, actor_rnn=False, env_model=None, device=None, rnn_mixer=False):
        assert conf["num_om_layers"] >= 1, "least have 1 layer opponent model"
        super(PPO_OM_MH, self).__init__(a_n_state=conf["n_state"] + sum(conf["n_opponent_action"]) if args.true_prob else conf["n_state"] + conf["opponent_model_hidden_layers"][-1],
                                        v_n_state=conf["n_state"],
                                        n_action=conf["n_action"],
                                        a_hidden_layers=conf["a_hidden_layers"],
                                        v_hidden_layers=conf["v_hidden_layers"],
                                        actor_rnn=actor_rnn,
                                        args=args,
                                        conf=conf,
                                        name="MBAM_" + name,
                                        logger=logger)
        self.agent_idx = agent_idx      #agent_index in env
        self.env_model = env_model
        self.oppo_model = Opponent_Model_MH(args, conf, self.name, device)
        self.om_buffer = OM_MH_Buffer(args, conf, device)
        self.om_phis = np.array([self.oppo_model.get_parameter()] * conf["num_om_layers"])
        if rnn_mixer:
            self.mix_ratio = torch.Tensor([1.0 / conf["num_om_layers"]] * conf["num_om_layers"]).to(device)
            if args.only_use_last_layer_IOP:
                temp = [0.0] * conf["num_om_layers"]
                temp[-1] = 1.0
                self.mix_ratio = torch.Tensor(temp).to(device)
        else:
            self.mix_ratio = np.array([1.0 / conf["num_om_layers"]] * conf["num_om_layers"])
            if args.only_use_last_layer_IOP:
                temp = [0.0] * conf["num_om_layers"]
                temp[-1] = 1.0
                self.mix_ratio = np.array(temp)
        self.device = device
        self.rnn_mixer = rnn_mixer
        if device is not None:
            self.change_device(device)

    def change_om_layers(self, num, rnn_mixer=False):
        self.conf["num_om_layers"] = num
        self.om_phis = np.array([self.oppo_model.get_parameter()] * self.conf["num_om_layers"])
        self.rnn_mixer = rnn_mixer
        if self.rnn_mixer:
            self.mix_ratio = torch.Tensor([1.0 / self.conf["num_om_layers"]] * self.conf["num_om_layers"]).to(self.device)
            if self.args.only_use_last_layer_IOP:
                temp = [0.0] * self.conf["num_om_layers"]
                temp[-1] = 1.0
                self.mix_ratio = torch.Tensor(temp).to(self.device)
        else:
            self.mix_ratio = np.array([1.0 / self.conf["num_om_layers"]] * self.conf["num_om_layers"])
            if self.args.only_use_last_layer_IOP:
                temp = [0.0] * self.conf["num_om_layers"]
                temp[-1] = 1.0
                self.mix_ratio = np.array(temp)

    def change_device(self, device):
        super(MBAM_OM_MH, self).change_device(device)
        self.oppo_model.change_device(device)

    def learn(self, data, iteration, no_log=False):
        super(MBAM_OM_MH, self).learn(data, iteration, no_log)

    def single_learn(self, data, iteration, no_log=False):
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
        """
        state
        (dcn(action).astype(np.int32),
                dcn(logp_a),
                dcn(entropy),
                dcn(value),
                dcn(action_prob),
                dcn(hidden_prob),
                dcn(hidden_state) if self.actor_rnn else None)
                reward: numpy [trj_length]
                last_val: numpy [1]
                reward -> advantage -> reward_to_go
        """

        def discount_cumsum(x, discount):
            """
                discount float
                x torch.Tensor  1dim [trj_size]
                all for torch.Tensor keep grad, for online update
                input:
                    vector x,
                    [x0,
                     x1,
                     x2]
                output:
                    [x0 + discount * x1 + discount^2 * x2,
                     x1 + discount * x2,
                     x2]
            """
            pass
            trj_size = x.shape[0]
            discount_list = torch.Tensor([discount ** i for i in range(trj_size)]).to(x.device)
            output = []
            for i in range(trj_size):
                if i == 0:
                    output.append(torch.sum(x * discount_list))
                else:
                    output.append(torch.sum(x[i:] * discount_list[:-i]))
            output = torch.stack(output)
            return output
            pass

        for param_group in self.a_optimizer.param_groups:
            param_group['lr'] = self.conf["a_learning_rate"]
        for param_group in self.v_optimizer.param_groups:
            param_group['lr'] = self.conf["v_learning_rate"]
        logp_a = data["logp_a"]
        value = data["value"]
        reward = data["reward"]
        last_val = data["last_val"]

        logp_a = torch.stack(logp_a)
        value = torch.stack(value)
        reward = np.stack(reward)

        reward_l = torch.from_numpy(reward).float().to(self.device)
        reward_l = torch.cat([reward_l, last_val], dim=0)

        value_l = torch.cat([value.view(-1), last_val], 0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = reward_l[:-1] + self.conf["gamma"] * value_l[1:] - value_l[:-1]
        advantage = discount_cumsum(deltas, self.conf["gamma"] * self.conf["lambda"])
        # the next line computes rewards-to-go, to be targets for the value function
        reward_to_go = discount_cumsum(reward_l, self.conf["gamma"])[:-1]

        adv_mean, adv_std = torch.mean(advantage), torch.std(advantage)
        if adv_std == 0.0:
            advantage = (advantage - adv_mean)
            adv_mean = adv_mean.detach().cpu()
            adv_std = adv_std.detach().cpu()
        else:
            advantage = (advantage - adv_mean) / adv_std
            adv_mean = adv_mean.detach().cpu()
            adv_std = adv_std.detach().cpu()

        """update a_net"""
        self.a_optimizer.zero_grad()
        loss_a = -torch.mean(logp_a * advantage.detach())
        loss_a.backward()
        self.a_optimizer.step()
        """update v_net"""
        self.v_optimizer.zero_grad()
        loss_v = torch.mean(((value - reward_to_go) ** 2))
        loss_v.backward()
        self.v_optimizer.step()

        if not no_log:
            loss_a_old = loss_a.detach().item()
            loss_v_old = loss_v.detach().item()
            self.logger.log_performance(tag=self.name + "/single", iteration=iteration,
                                        Loss_a=loss_a_old, Loss_v=loss_v_old,
                                        )
        pass

    def single_choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
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
        raise NotImplementedError       #just for rnn_mixer
        assert type(state) is np.ndarray or type(state) is torch.Tensor, "choose_action input type error"

        if type(state) is np.ndarray:
            state = torch.Tensor(state).view(-1, self.conf["n_state"]).to(device=self.device)
        if oppo_hidden_prob is None:
            if self.actor_rnn:
                raise NotImplementedError
                oppo_hidden_prob = self._get_mixed_om_hidden_prob(state, hidden_state=hidden_state)
            else:
                oppo_hidden_prob = self._get_mixed_om_hidden_prob(state)
        if self.actor_rnn:
            raise NotImplementedError
            return super(MBAM_OM_MH, self).single_choose_action(state, greedy, oppo_hidden_prob=oppo_hidden_prob, hidden_state=hidden_state)
        return super(MBAM_OM_MH, self).single_choose_action(state, greedy, oppo_hidden_prob=oppo_hidden_prob)

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
        if oppo_hidden_prob is None:
            if self.actor_rnn:
                raise NotImplementedError
                oppo_hidden_prob = self._get_mixed_om_hidden_prob(state, hidden_state=hidden_state)
            else:
                oppo_hidden_prob = self._get_mixed_om_hidden_prob(state)
            self.oppo_hidden_prob = [oppo_hidden_prob[i].clone().detach().cpu() for i in range(len(oppo_hidden_prob))]
        if self.actor_rnn:
            raise NotImplementedError
            return super(MBAM_OM_MH, self).choose_action(state, greedy, oppo_hidden_prob=oppo_hidden_prob, hidden_state=hidden_state)

        # multi-head opponent hidden prob -> single head opponent hidden prob
        if type(oppo_hidden_prob[0]) is np.ndarray:
            oppo_hidden_prob = np.concatenate(oppo_hidden_prob, axis=1)
        elif type(oppo_hidden_prob[0]) is torch.Tensor:
            oppo_hidden_prob = torch.cat(oppo_hidden_prob, dim=1)
        else:
            raise TypeError
        return super(MBAM_OM_MH, self).choose_action(state, greedy, oppo_hidden_prob=oppo_hidden_prob)

    def _get_mixed_om_hidden_prob(self, state, hidden_state=None):
        '''
        :param state: np.ndarry  shape is (n_batch, n_state)
        :return: mixed_om_hidden_prob: np.ndarry  shape is (n_batch, n_hidden_prob)
        '''
        if type(state) is np.ndarray:
            state = torch.Tensor(state).to(self.device)
        if type(hidden_state) is np.ndarray:
            hidden_state = torch.Tensor(hidden_state).to(self.device)
        if self.actor_rnn:
            self._gen_om_phis(state, hidden_state=hidden_state)
        else:
            self._gen_om_phis(state)
        if self.rnn_mixer:
            raise NotImplementedError
            actions_om_probs = []
            with torch.no_grad():
                for k in range(0, self.conf["num_om_layers"]):
                    actions_om_probs_k, _ = self.oppo_model.get_action_prob(state, self.om_phis[k])
                    actions_om_probs.append(actions_om_probs_k)
                actions_om_probs = torch.cat(actions_om_probs, dim=0)
            mixed_om_hidden_probs = torch.sum(actions_om_probs * self.mix_ratio.view(-1, 1), dim=0)
        else:
            mixed_phis = soft_update(self.om_phis, self.mix_ratio)
            with torch.no_grad():
                mixed_om_action_probs, mixed_om_hidden_probs = self.oppo_model.get_action_prob(state, mixed_phis)
        print("duishou",mixed_om_hidden_probs)
        return mixed_om_hidden_probs
    def _gen_om_phis(self, state, hidden_state=None):
        #print("\n\nstart rollout")
        #print(state)
        for k in range(1, self.conf["num_om_layers"]):
            if self.actor_rnn:
                if not self.args.random_best_response:
                    best_response = self._rollout(state, self.om_phis[k - 1], hidden_state=hidden_state)
                else:
                    #best_response = np.random.choice(np.arange(self.conf["n_opponent_action"]))
                    best_response = torch.randint(low=0, high=self.conf["n_opponent_action"], size=(1, 1)).long()
                #print("best response is ", best_response)
                #print("type of best response is ", type(best_response))
            else:
                if not self.args.random_best_response:
                    best_response = self._rollout(state, self.om_phis[k - 1])
                else:
                    best_response = [torch.randint(low=0, high=self.conf["n_opponent_action"][i], size=(1, 1)).long() for i in range(len(self.conf["n_opponent_action"]))]
                #print("layer: {}, best response is {}".format(k,best_response))
            data = dict({"state": state, "action": best_response})
            phi, loss = self.oppo_model.learn(data, self.om_phis[k - 1],
                                               lr=self.conf["imagine_model_learning_rate"],
                                               l_times=self.conf["imagine_model_learning_times"])
            self.om_phis[k] = phi
        return None

    def _rollout(self, state, om_phi, hidden_state=None):
        '''
        :param state: np.ndarray, [1, n_state]
                    -----> [n_batch, n_state]
        :param om_phi: np.ndarray, [1]
        :return: best_response, torch.Tensor, int, [1, 1]
        这里采用MC-tree search
        向下搜索roll_out_length层
        每一层扩展 n_opponent_action 倍
        在扩展时, state采用repeat_interleave的方式,逐个复制
                 opponent_action采用repeat的方式,逐支复制
        共计有 n_opponent_action**roll_out_length个子节点
        '''
        assert type(state) is torch.Tensor, "_rollout input type error"

        max_batch = self.conf["n_opponent_action"][0] ** self.conf["roll_out_length"]

        rew_record = None
        done_record = None
        cur_batch = 1
        self.env_model.reset()
        #oppo_action = torch.LongTensor([i for i in range(self.conf["n_opponent_action"])]).view(-1, 1)
        SIMPLE_TAG_n_action = self.conf["n_opponent_action"]
        oppo_action = np.arange(SIMPLE_TAG_n_action[0]).reshape(-1, 1)
        for i in range(1, len(SIMPLE_TAG_n_action)):
            oppo_action = np.concatenate([np.repeat(oppo_action, SIMPLE_TAG_n_action[i], axis=0),
                                   np.tile(np.arange(SIMPLE_TAG_n_action[i]).reshape(-1, 1), [oppo_action.shape[0], 1])], axis=1)
        oppo_action_space_size = oppo_action.shape[0]
        oppo_action = torch.from_numpy(oppo_action).long()
        oppo_action_origin = oppo_action.clone()
        for i in range(self.conf["roll_out_length"]):
            if i != 0:
                oppo_action = oppo_action.repeat(self.conf["n_opponent_action"], 1)  # [next_batch, 1]

            cur_batch = cur_batch * self.conf["n_opponent_action"]

            state = state.repeat_interleave(oppo_action_space_size, dim=0)
            if self.actor_rnn:
                hidden_state = hidden_state.repeat_interleave(oppo_action_space_size, dim=0)

            with torch.no_grad():
                # get oppo_hidden_prob
                om_action_prob, om_hidden_prob = self.oppo_model.get_action_prob(state, om_phi)
                # my policy->action
                action, _, _, _, _, _, hidden_state = self.choose_action(state, greedy=True, oppo_hidden_prob=om_hidden_prob, hidden_state=hidden_state) #[cur_batch, 1]
            #print("om prob is", om_hidden_prob[0])
            if self.actor_rnn:
                hidden_state = torch.Tensor(hidden_state).to(device=self.device)

            # env_model get next_state and reward
                


            actions = [None, None]
            actions[self.agent_idx] = action
            actions[1 - self.agent_idx] = oppo_action
            state_, reward, done = self.env_model.step(state, actions)   #[cur_batch, n_state]
            #print("action is {}".format(action))
            # record reward and done
            if rew_record is None:
                rew_record = reward[self.agent_idx].view(-1, 1).to(self.device)                 # [cur_batch, 1]
                done_record = done.view(-1, 1).to(self.device)                                # [cur_batch, 1]
            else:
                rew_record = torch.cat([rew_record, reward[self.agent_idx]], dim=1)     # [cur_batch, 1 --> roll_out_length]
                done_record = torch.cat([done_record, done], dim=1)     # [cur_batch, 1 --> roll_out_length]
            if i != self.conf["roll_out_length"] - 1:
                rew_record = rew_record.repeat(self.conf["n_opponent_action"], 1)
                done_record = done_record.repeat(self.conf["n_opponent_action"], 1)
            # prepare for next step
            state = state_
        ''' 
                        done矩阵 行代表tree-search-branch,列代表序列reward
                        目的使reward矩阵添加上value,以计算discount reward
                        [r0, r1, r2, r3, r4]    -->   [r0, r1, r2, r3, r4, v]
                        因为有可能提前结束, 也有可能 -->   [r0, r1, r2, v, 0, 0]
                        方法如下:
                        0 0 0 0 1       (0)  原始done矩阵                   r, r, r, r, r       (00)  原始reward矩阵
                        0 0 0 1 1                                          r, r, r, r, e 
                        0 1 1 1 1                                          r, r, e, e, e
                        0 0 0 0 0                                          r, r, r, r, r
                        1 1 1 1 1                                          r, e, e, e, e

                        0 0 0 0 0 1     (1)  将(0)右移,左填0                   r, r, r, r, r     (11)  (00) * 1-(1)[0:5]
                        0 0 0 0 1 1                                          r, r, r, r, 0            (00)-(1)的前5列取反
                        0 0 1 1 1 1                                          r, r, 0, 0, 0
                        0 0 0 0 0 0                                          r, r, r, r, r
                        0 1 1 1 1 1                                          r, 0, 0, 0, 0

                        0 0 0 0 0 0     (2)  将(1)右移,左填0
                        0 0 0 0 0 1
                        0 0 0 1 1 1
                        0 0 0 0 0 0
                        0 0 1 1 1 1

                        0 0 0 0 0 1     (3)  (1)-(2)
                        0 0 0 0 1 0
                        0 0 1 0 0 0
                        0 0 0 0 0 0
                        0 1 0 0 0 0

                        0 0 0 0 0 0     (4)  (3)[0:5] + 1-(0)[4]             r, r, r, r, r, 0   (result)  (11) +  {1-(0)[4]} * v
                        0 0 0 0 1 0          (3)的前5列+(0)的最后一列取反       r, r, r, r, 0, 0             (11) + v*(0)的最后一列取反 
                        0 0 1 0 0 0                                          r, r, 0, 0, 0, 0
                        0 0 0 0 0 1                                          r, r, r, r, r, v
                        0 1 0 0 0 0                                          r, 0, 0, 0, 0, 0
        '''
        with torch.no_grad():
            v = self.v_net(state_)
            v = torch.zeros_like(v)
        left_zero = torch.zeros((done_record.shape[0], 1), device=self.device)
        temp_1 = torch.cat([left_zero.bool(), done_record], dim=1)
        temp_11 = rew_record * (~temp_1[:, :-1])
        reward_record = torch.cat([temp_11.float(), v * (~done_record[:, -1:])], dim=1)

        if not hasattr(self, "gamma_list"):
            self.gamma_list = torch.tensor([pow(self.conf["gamma"], i) for i in range(self.conf["roll_out_length"] + 1)], device=self.device)
        discount_r = torch.sum(reward_record * self.gamma_list, axis=1)
        best_response = (torch.argmin(discount_r, dim=0) / (oppo_action_space_size ** (self.conf["roll_out_length"] - 1))).view(1, 1).long()
        best_response = oppo_action_origin[best_response.item()].view(-1, 1, 1).to(device=self.device)
        #best_response = (torch.argmax(discount_r, dim=0) / (self.conf["n_opponent_action"] ** (self.conf["roll_out_length"] - 1))).view(1, 1).long()   # cooper
        #print("best response is ", best_response)
        return best_response

    def observe_oppo_action(self, state, oppo_action, iteration, no_log=True):
        '''
        :param state: np.ndarray, [1, n_state]
        :param oppo_action: np.ndarray or int [1]
        :return: None
        '''
        assert type(state) is np.ndarray, "observe_oppo_action input type error"
        #assert (type(oppo_action) is np.ndarray) or (type(oppo_action) is int), "observe_oppo_action input type error"
        #assert state.shape == (1, self.conf["n_state"]), "observe_oppo_action input shape error"

        if type(oppo_action[0]) is not np.ndarray:
            oppo_action = [np.array([oppo_action[i]]) for i in range(len(self.conf["n_opponent_action"]))]
        oppo_action = [oppo_action[i].reshape(1, 1) for i in range(len(self.conf["n_opponent_action"]))]
        #assert oppo_action.shape == (1, 1), "observe_oppo_action input shape error"
        state = state.reshape(1, -1)
        state = torch.Tensor(state).to(device=self.device)
        oppo_action = [torch.LongTensor(oppo_action[i]).to(device=self.device) for i in range(len(self.conf["n_opponent_action"]))]
        # cal mix ratio
        if self.conf["num_om_layers"] > 1:
            if self.rnn_mixer:
                raise NotImplementedError
                self._rnn_cal_mix_ratio(state, oppo_action)
                mix_ratio = self.mix_ratio.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    self._cal_mix_ratio(state, oppo_action)
                    mix_ratio = self.mix_ratio
        # store experience
        self.om_buffer.store_memory(state, oppo_action)

        # learn model phi_0
        loss = 0
        if self.om_buffer.size > 2 * self.conf["opponent_model_batch_size"]:
            data = self.om_buffer.get_batch(self.conf["opponent_model_batch_size"])
            phi, loss = self.oppo_model.learn(data, self.om_phis[0],
                                              lr=self.conf["opponent_model_learning_rate"],
                                              l_times=self.conf["opponent_model_learning_times"])
            self.om_phis[0] = phi

        if not no_log:
            self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, Loss_oppo=loss)
            if self.conf["num_om_layers"] > 1:
                mix_ratio_record = {"om_layer{}_mix_ratio".format(i): mix_ratio[i] for i in range(len(mix_ratio))}
                self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, **mix_ratio_record)
                self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, Mix_entropy=torch.distributions.Categorical(torch.Tensor(mix_ratio)).entropy())
        pass
    def _cal_mix_ratio(self, state, oppo_action):
        '''
        :param state: torch.Tensor, [1, n_state]
        :param oppo_action: torch.Tensor, [1, 1]
        :return: None
        '''
        '''
        om_action_prob = P(model|action)
        action_om_prob = P(action|model)
        om_prob = P(model)
        actionom_prob = P(action, model) 
        sum_om_action_prob: decay sum om_action_prob
        '''
        if not hasattr(self, "short_term_decay_list"):
            self.short_term_decay_list = torch.tensor([pow(self.conf["short_term_decay"], i) for i in range(self.conf["short_term_horizon"])], device=self.device)
            self.all_om_action_prob = torch.full((self.conf["num_om_layers"], self.conf["short_term_horizon"]), fill_value=(1.0 / self.conf["num_om_layers"]), device=self.device)  # [num_om_layers, error_horizon]
        actions_om_probs = [[] for _ in range(len(self.conf["n_opponent_action"]))]
        om_prob = self.all_om_action_prob.mean(dim=1)
        # cal each phi's action prob
        with torch.no_grad():
            for k in range(0, self.conf["num_om_layers"]):
                actions_om_probs_k, _ = self.oppo_model.get_action_prob(state, self.om_phis[k])
                [actions_om_probs[i].append(actions_om_probs_k[i]) for i in range(len(self.conf["n_opponent_action"]))]
        actions_om_probs = [torch.cat(actions_om_probs[i], dim=0) for i in range(len(self.conf["n_opponent_action"]))]
        action_om_probs = [actions_om_probs[i][:, oppo_action[i].squeeze()] for i in range(len(self.conf["n_opponent_action"]))]
        temp_probs = action_om_probs[0]
        for i in range(1, len(self.conf["n_opponent_action"])):
            temp_probs = temp_probs * action_om_probs[i]
        actionom_prob = temp_probs * om_prob
        om_action_prob = actionom_prob / actionom_prob.sum(dim=0, keepdim=True).repeat(self.conf["num_om_layers"])
        self.all_om_action_prob = torch.cat([om_action_prob.unsqueeze(dim=1), self.all_om_action_prob[:, :-1]], dim=1)
        # cal each layers prob
        sum_om_action_prob = (self.all_om_action_prob * self.short_term_decay_list).sum(dim=1)
        # cal_mix_ratio
        self.mix_ratio = lambda_softmax(sum_om_action_prob, dim=0, factor=self.conf["mix_factor"]).cpu().numpy()
        pass
    def _rnn_cal_mix_ratio(self, state, oppo_action):
        '''
        :param state: torch.Tensor, [1, n_state]
        :param oppo_action: torch.Tensor, [1, 1]
        :return: None
        '''
        if not hasattr(self, "mixer"):
            from base.Actor_RNN import Actor_RNN
            import itertools
            self.mixer = Actor_RNN(input=self.conf["num_om_layers"],
                                   output=self.conf["num_om_layers"],
                                   hidden_layers_features=[8, 8],
                                   output_type="prob").to(self.device)
            self.mixer_hidden = torch.zeros(size=(1, 8)).to(self.device)
            self.a_optimizer = torch.optim.Adam(itertools.chain(self.a_net.parameters(), self.mixer.parameters()), lr=self.conf["a_learning_rate"])
        actions_om_probs = []
        with torch.no_grad():
            for k in range(0, self.conf["num_om_layers"]):
                actions_om_probs_k, _ = self.oppo_model.get_action_prob(state, self.om_phis[k])
                actions_om_probs.append(actions_om_probs_k)
            actions_om_probs = torch.cat(actions_om_probs, dim=0)
            target = oppo_action.squeeze().repeat(self.conf["num_om_layers"])
            error_list = torch.nn.functional.cross_entropy(actions_om_probs, target, reduce=False)

        ratio, _, hidden = self.mixer(error_list.view(1, -1), self.mixer_hidden)
        self.mixer_hidden = hidden
        self.mix_ratio = ratio.squeeze()
        #print(list(self.mixer.parameters()))
        #self.mix_ratio, _, self.mixer_hidden = self.mixer(error_list.view(1, -1), self.mixer_hidden)
        #self.mix_ratio = self.mix_ratio.squeeze()
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
        torch.save(obj, filepath, _use_new_zipfile_serialization=False)
        self.logger.log("model saved in {}".format(filepath))

    @staticmethod
    def load_model(filepath, args, logger, device, **kwargs):
        assert "env_model" in kwargs.keys(), "must input env_model"
        checkpoint = torch.load(filepath, map_location='cpu')
        #args = checkpoint["args"]
        conf = checkpoint["conf"]
        name = checkpoint["name"].replace("MBAM_", "")
        agent_idx = checkpoint["agent_idx"]
        actor_rnn = checkpoint["actor_rnn"]
        mbam = MBAM_OM_MH(args, conf, name, logger, agent_idx, actor_rnn, kwargs["env_model"], device, rnn_mixer=args.rnn_mixer)

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
def lambda_softmax(y, dim, factor):
    '''
    :param y: torch.Tensor
    :param dim:
    :param factor:
    :return: softmax
    normalize y - y.mean
    x = e * factor
    pi = x^yi/sum(x^yj) for i
    '''
    import math
    with torch.no_grad():
        x = math.e * factor
        y = y - y.mean(dim=dim, keepdim=True)
        t = torch.pow(x, y)
        sum_t = t.sum(dim=dim, keepdim=True).repeat_interleave(repeats=y.shape[dim] ,dim=dim)
        t = t / sum_t
        return t

if __name__ == "__main__":
    import time
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_name", type=str, default="football_ppo_vs_mbam", help="football_ppo_vs_mbam\n " +
                                                                                     "predator_ppo_vs_mbam\n" +
                                                                                     "rps_ppo_vs_mbam\n" +
                                                                                     "keepaway_ppo_vs_mbam\n" +
                                                                                     "trigame_ppo_vs_mbam\n" +
                                                                                     "simple_rps_ppo_vs_mbam")
    parser.add_argument("--env", type=str, default="football", help="football\n" +
                                                                    "predator\n" +
                                                                    "rps\n" +
                                                                    "keepaway\n" +
                                                                    "trigame\n" +
                                                                    "simple_rps")
    parser.add_argument("--prefix", type=str, default="test", help="train or test or search")
    parser.add_argument("--train_mode", type=int, default=0, help="0 1 means ppovsmbam and mbamvsppo")
    parser.add_argument("--alter_train", type=int, default=0, help="0 1 means no and yes")
    parser.add_argument("--alter_interval", type=int, default=100, help="epoch")
    parser.add_argument("--test_mode", type=int, default=1, help="0 1 2 means layer0, layer1, layer2")
    parser.add_argument("--test_mp", type=int, default=1, help="multi processing")

    parser.add_argument("--seed", type=int, default=-1, help="-1 means random seed")
    parser.add_argument("--ranks", type=int, default=1, help="for prefix is train")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--dir", type=str, default="", help="")

    parser.add_argument("--eps_max_step", type=int, default=30, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=10, help="")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="")
    parser.add_argument("--max_epoch", type=int, default=100, help="train epoch")
    parser.add_argument("--num_om_layers", type=int, default=3, help="train epoch")
    parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")

    parser.add_argument("--actor_rnn", type=bool, default=True, help="True or False")
    parser.add_argument("--true_prob", type=bool, default=True, help="True or False")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--policy_training", type=bool, default=True, help="True or False")
    args = parser.parse_args()
    starttime = time.time()

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
        "num_om_layers": 3,
        "opponent_model_hidden_layers": [64, 32],
        "opponent_model_memory_size": 1000,
        "opponent_model_learning_rate": 0.001,
        "opponent_model_batch_size": 64,
        "opponent_model_learning_times": 1,

        "imagine_model_learning_rate": 0.001,
        "imagine_model_learning_times": 5,
        # times is 0 means training until optimal action is max prob , but unfinished
        "roll_out_length": 5,
        "short_term_decay": 0.9,  # error discount factor
        "short_term_horizon": 10,
        "mix_factor": 0.01,  # adjust mix cuvre, Strongly related to error_horizon and error_delay

        # ppo setting
        "v_hidden_layers": [64, 32],
        "a_hidden_layers": [64, 32],
        "v_learning_rate": 0.001,
        "a_learning_rate": 0.001,
        "gamma": 0.99,  # value discount factor
        "lambda": 0.99,  # general advantage estimator
        "epsilon": 0.115,  # ppo clip param
        "entcoeff": 0.0015,
        "a_update_times": 10,
        "v_update_times": 10,
        # ppo buffer setting
        "buffer_memory_size": 300,  # update_episode * max_episode_step = 20 * 30
    }
    BATCH_SIZE = 1
    # def env_model(state, actions, reward_idx = 1):
    #     shape = state.shape
    #     state_ = torch.Tensor(np.random.random(shape)).to(device=args.device)
    #     reward = torch.Tensor(np.random.random((shape[0], 1))).to(device=args.device)
    #     done = torch.Tensor([False]).view(1, 1).repeat(shape[0], 1).to(device=args.device)
    #     return state_, reward, done
    # def train(model):
    #     BATCH_SIZE = 1
    #     for i in range(10):
    #         state = np.random.random((BATCH_SIZE, 24))
    #         hidden_prob = np.random.random((BATCH_SIZE, 16))
    #         action, logp_a, entropy, value, action_prob, hidden_prob, hidden_state = mbam.choose_action(state,
    #                                                                                                     greedy=False,
    #                                                                                                     oppo_hidden_prob=None,
    #                                                                                                     hidden_state=hidden_state)
    #         print("choose action", i)
    #         state = np.random.random((1, 24))
    #         oppo_action = np.random.randint(0, 10, (1, 1))
    #         mbam.observe_oppo_action(state, oppo_action)
    #         print(mbam.mix_ratio)
    #         print("observe_oppo_action", i)
    #     print("learn test")
    #     for i in range(10000):
    #         BATCH_SIZE = np.random.randint(300)
    #         state = np.random.random((BATCH_SIZE, 24))
    #         action = np.random.randint(0, 10, (BATCH_SIZE, 1))
    #         reward = np.random.random((BATCH_SIZE, 1))
    #         value = np.random.random((BATCH_SIZE, 1))
    #         logp_a = np.random.random((BATCH_SIZE, 1))
    #         hidden_state = np.random.random((BATCH_SIZE, 32))
    #         oppo_hidden_prob = np.random.random((BATCH_SIZE, 16))
    #         data = dict({"state": state,
    #                      "action": action,
    #                      "reward": reward,
    #                      "value": value,
    #                      "logp_a": logp_a,
    #                      "hidden_state": hidden_state,
    #                      "oppo_hidden_prob": oppo_hidden_prob, })
    #         ppobf.store_memory(data, last_val=0)
    #         print(ppobf.next_idx)
    #         if ppobf.next_idx > 500:
    #             data = ppobf.get_batch()
    #             model.learn(data, 100, no_log=True)
    # mbam = MBAM(args=args, conf=conf, name="AAA", logger=None, agent_idx=1, actor_rnn=True, env_model=env_model, device=args.device)
    # hidden_state = mbam.init_hidden_state(BATCH_SIZE)
    # mbam.share_memory()
    # train(mbam)
    # # from DRL_MARL_homework.MBAM_v2.baselines.PPO import PPO_Buffer
    # # ppobf = PPO_Buffer(args, conf, mbam.name, True, "cuda")
    # # import multiprocessing as mp
    # # mp.set_start_method("spawn")
    # # processes = []
    # # for rank in range(4):
    # #     p = mp.Process(target=train, args=(mbam,))
    # #     p.start()
    # #     processes.append(p)
    # # for p in processes:
    # #     p.join()
    # # print("end")
    #
    # endtime = time.time()
    # print("time", endtime - starttime)
    from env_wapper.football_penalty_kick.football_1_vs_1_penalty_kick import make_env as football_env
    from env_model.football_penalty_kick.model_football_1_vs_1_penalty_kick import load_env_model as football_env_model
    from env_model.football_penalty_kick.model_football_1_vs_1_penalty_kick import ENV_football_1_vs_1_penalty_kick
    env = football_env()
    env_model = football_env_model(args.device)
    mbam = MBAM(args=args, conf=conf, name="AAA", logger=None, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model,
                device=args.device, rnn_mixer=True)

    for eps in range(100):
        print("episode {} start".format(eps))
        state = env.reset()
        logp_a_l = []
        value_l = []
        reward_l = []
        last_value = None
        while True:
            hidden_state = mbam.init_hidden_state(n_batch=1)
            action, logp_a, entropy, value, action_prob, hidden_prob, hidden_state = mbam.single_choose_action(state[1],
                                                                                                        greedy=False,
                                                                                                        oppo_hidden_prob=None,
                                                                                                        hidden_state=hidden_state)
            oppo_a = np.random.randint(0, 10, (1),dtype=int)
            actions = [action.item(), oppo_a.item()]
            state_, reward, done, info = env.step(actions)

            mbam.observe_oppo_action(state[1], oppo_a, iteration=eps, no_log=True)

            reward_l.append(reward[1])
            logp_a_l.append(logp_a)
            value_l.append(value)
            state = state_

            if done:
                last_value = mbam.v_net(torch.from_numpy(state[1]).to(mbam.device))
                """train"""
                mbam.single_learn(data={"logp_a": logp_a_l,
                                        "value": value_l,
                                        "reward": reward_l,
                                        "last_val": last_value}, iteration=eps, no_log=True)
                break