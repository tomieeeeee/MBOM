import numpy as np
import torch
from memory_profiler import profile
def discount_cumsum(x, discount):
    import scipy.signal
    """
    magic from rllab for computing discounted cumulative sums of vectors.
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
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
import multiprocessing as mp

class Episode_Memory():
    def __init__(self):
        self.next_idx = 0
        self.state = []
        self.action = []
        self.logp_a = []
        self.entropy = []
        self.value = []
        self.oppo_hidden_prob = []
        self.hidden_state = []
        self.reward = []

        self.final_state = None
        pass

    def store_final_state(self, state, info):
        self.final_state = (state, info)
    def store_oppo_logp_a(self, oppo_logp_a):
        if not hasattr(self, "oppo_logp_a"):
            self.oppo_logp_a = []
        self.oppo_logp_a.append(oppo_logp_a)
    def store_env_info(self, state, reward):
        self.state.append(state)
        self.reward.append(reward)
    def store_oppo_hidden_prob(self, oppo_hidden_prob):
        self.oppo_hidden_prob.append(oppo_hidden_prob)
    def store_action_info(self, choose_action_return):
        #action, logp_a, entropy, value, action_prob, hidden_prob, hidden_state
        self.action.append(choose_action_return[0])
        self.logp_a.append(choose_action_return[1])
        self.entropy.append(choose_action_return[2])
        self.value.append(choose_action_return[3])
        self.hidden_state.append(choose_action_return[6])
    def get_data(self, is_meta_mapg=False):
        data = dict({
            "state":np.stack(self.state),
            "action":np.stack(self.action),
            "logp_a":torch.stack(self.logp_a) if is_meta_mapg else np.stack(self.logp_a),
            "entropy":np.stack(self.entropy),
            "value":np.stack(self.value).squeeze(axis=1),
            "reward":np.stack(self.reward).reshape(-1, 1),
            "oppo_hidden_prob":np.stack(self.oppo_hidden_prob).squeeze(axis=1),
            "hidden_state":np.stack(self.hidden_state).squeeze(axis=1) if np.any(np.stack(self.hidden_state) != None) else np.stack(self.hidden_state),
        })
        if hasattr(self, "oppo_logp_a"):
            data["oppo_logp_a"] = np.stack(self.oppo_logp_a)
        return data
    def get_score(self):
        score = 0
        for i in range(len(self.reward)):
            score += self.reward[i]
        return float(score)

def collect_trajectory(agents, env, args, global_step, is_prophetic=False, greedy=False):
    """Collect a epoch trajectory
    :param agents: list, [agent0, agent1]
    :param env:
    :param args: contains eps_per_epoch
    :return memories: list, [2, eps_per_epoch] contains Memory
            scores: list, [2], return epoch ave reward
    """
    memories = [[], []]
    scores = [[], []]
    for _ in range(1, args.eps_per_epoch + 1):
        # Initialize LSTM state
        hidden_state = [agent.init_hidden_state(n_batch=1) for agent in agents]
        oppo_hidden_prob = np.array([None, None])
        # Begin to collect trajectory
        state = env.reset()
        temp_memory = [Episode_Memory(), Episode_Memory()]
        while True:
            global_step += 1
            actions = np.array([0, 0], dtype=int)
            for agent_idx, agent in enumerate(agents):
                if type(agent).__name__ == "MBAM":
                    '''
                     action, np.ndarray int32 (num_trj, 1, 1)
                     logp_a, np.ndarray float (num_trj, 1, 1)
                     entropy, np.ndarray float (num_trj, 1, 1)
                     value, np.ndarray float (num_trj, 1, 1)
                     action_prob, np.ndarray float (num_trj, 1, n_action)
                     hidden_prob, np.ndarray float (num_trj, 1, n_hidden_prob), this is actor network number of latest layer'cell
                     hidden_state, np.ndarray float (num_trj, 1, n_hidden_state), is None if not actor_rnn
                    '''
                    #print("agent{} start choose_action, layers is {}".format(agent.agent_idx, agent.conf["num_om_layers"]))
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1-agent_idx] if is_prophetic == True else None, greedy=greedy, global_step=global_step)
                    #temp_memory[agent_idx].mixed_action_prob = mixed_action_prob

                elif type(agent).__name__ == "PPO":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=None, greedy=greedy)
                elif type(agent).__name__ == "Meta_mapg":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx])
                    # store oppo_log_a
                    temp_memory[agent_idx].store_oppo_logp_a(action_info[7])
                elif type(agent).__name__ == "Fake_MBAM":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1-agent_idx] if is_prophetic == True else None, greedy=greedy)
                else:
                    raise TypeError

                if args.prophetic_onehot:
                    temp_a = action_info[0].item()
                    oppo_hidden_prob[agent_idx] = np.eye(action_info[5].shape[1])[temp_a].reshape(1, -1)
                    action_info = (action_info[0], action_info[1], action_info[2], action_info[3], action_info[4], oppo_hidden_prob[agent_idx], action_info[6])
                else:
                    oppo_hidden_prob[agent_idx] = action_info[5]
                # store action info
                temp_memory[agent_idx].store_action_info(action_info)
                # store oppo hidden prob
                temp_memory[1 - agent_idx].store_oppo_hidden_prob(action_info[5])
                # record
                hidden_state[agent_idx] = action_info[6]
                actions[agent_idx] = action_info[0].item()
            #env.render()
            # env interact

            """record some IOPs weight"""
            if args.record_more and args.prefix == "test":
                true_prob = temp_memory[1].oppo_hidden_prob[-1][0]
                true_action = actions[0].item()
                mix_parameter = agents[1].oppo_hidden_prob[0].numpy()
                layers_prob = []
                layers_action = []
                for i in range(args.num_om_layers):
                    layers_prob.append(agents[1].oppo_model.get_action_prob(state[1], agents[1].om_phis[i])[0].detach().cpu().numpy())
                    layers_action.append(np.argmax(layers_prob[i]))
                if args.rnn_mixer:
                    mix_prob = np.sum(np.array(layers_prob) * agents[1].mix_ratio.detach().cpu().numpy().reshape(-1,1), axis=0)
                else:
                    mix_prob = np.sum(np.array(layers_prob) * agents[1].mix_ratio.reshape(-1,1), axis=0)
                dis_mix_parameter = -np.sum(true_prob * np.log(mix_parameter))
                dis_mix_prob = -np.sum(true_prob * np.log(mix_prob))
                dis_layers_prob = {"Dis_layers_{}_prob".format(i): -np.sum(true_prob * np.log(layers_prob[i])) for i in range(args.num_om_layers)}

                hit_ratio_layers = {"Hit_ratio_{}".format(i): 1 if true_action == np.argmax(layers_prob[i]) else 0 for i in range(args.num_om_layers)}
                hit_ratio_mix_parameter = 1 if true_action == np.argmax(mix_parameter) else 0
                hit_ratio_mix_prob = 1 if true_action == np.argmax(mix_prob) else 0

                agents[1].logger.log_performance(tag=agents[1].name + "/steps", iteration=global_step,
                                                 Dis_mix_parameter=dis_mix_parameter, Dis_mix_prob=dis_mix_prob,
                                                 Hit_ratio_mix_parameter = hit_ratio_mix_parameter, Hit_ratio_mix_prob=hit_ratio_mix_prob,
                                                 **dis_layers_prob, **hit_ratio_layers)


            state_, reward, done, info = env.step(actions)

            if args.env == "coin_game" and info is not None:
                if info["same_pick"]:
                    agents[1].logger.log_performance(tag=agents[1].name + "/steps", iteration=global_step, Same_pick=1)
                else:
                    agents[1].logger.log_performance(tag=agents[1].name + "/steps", iteration=global_step, Same_pick=0)

            #print(reward)
            #print("actions is " ,actions)

            # store env info
            for i in range(len(agents)):
                temp_memory[i].store_env_info(state[i], reward[i])

            # store oppo action
            if not is_prophetic:
                for agent_idx, agent in enumerate(agents):
                    if hasattr(agent, "observe_oppo_action"):
                        agent.observe_oppo_action(state=state[agent_idx], oppo_action=actions[1-agent_idx],
                                                  iteration=global_step, no_log=False)
            state = state_

            if done:
                for i in range(len(agents)):
                    temp_memory[i].store_final_state(state_[i], info)
                    memories[i].append(temp_memory[i])
                    scores[i].append(temp_memory[i].get_score())
                agents[1].logger.log_performance(tag=agents[1].name + "/steps", iteration=global_step
                                                 #Same_pick_sum=info["same_pick_sum"],
                                                 #Coin_sum=info["coin_sum"],
                                                 #Pick_ratio=info["same_pick_sum"]/info["coin_sum"]
                                                )
                break
    scores = [sum(scores[i])/len(scores[i]) for i in range(len(agents))]
    return memories, scores, global_step

def collect_trajectory_reversed(agents, env, args, global_step, is_prophetic=False):
    """Collect a epoch trajectory
    :param agents: list, [agent0, agent1]
    :param env:
    :param args: contains eps_per_epoch
    :return memories: list, [2, eps_per_epoch] contains Memory
            scores: list, [2], return epoch ave reward
    """
    memories = [[], []]
    scores = [[], []]

    for _ in range(1, args.eps_per_epoch + 1):
        # Initialize LSTM state
        hidden_state = [agent.init_hidden_state(n_batch=1) for agent in agents]
        oppo_hidden_prob = np.array([None, None])
        # Begin to collect trajectory
        state = env.reset()
        temp_memory = [Episode_Memory(), Episode_Memory()]
        while True:
            global_step += 1
            actions = np.array([0, 0], dtype=int)
            for agent_idx, agent in list(enumerate(agents))[::-1]:
                if type(agent).__name__ == "MBAM":
                    '''
                     action, np.ndarray int32 (num_trj, 1, 1)
                     logp_a, np.ndarray float (num_trj, 1, 1)
                     entropy, np.ndarray float (num_trj, 1, 1)
                     value, np.ndarray float (num_trj, 1, 1)
                     action_prob, np.ndarray float (num_trj, 1, n_action)
                     hidden_prob, np.ndarray float (num_trj, 1, n_hidden_prob), this is actor network number of latest layer'cell
                     hidden_state, np.ndarray float (num_trj, 1, n_hidden_state), is None if not actor_rnn
                    '''
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1-agent_idx] if is_prophetic == True else None)
                elif type(agent).__name__ == "PPO":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=None)
                elif type(agent).__name__ == "Meta_mapg":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx])
                    # store oppo_log_a
                    temp_memory[agent_idx].store_oppo_logp_a(action_info[7])
                else:
                    raise TypeError
                if args.prophetic_onehot:
                    temp_a = action_info[0].item()
                    oppo_hidden_prob[agent_idx] = np.eye(action_info[5].shape[1])[temp_a].reshape(1, -1)
                    action_info = (action_info[0], action_info[1], action_info[2], action_info[3], action_info[4], oppo_hidden_prob[agent_idx], action_info[6])
                else:
                    oppo_hidden_prob[agent_idx] = action_info[5]
                # store action info
                temp_memory[agent_idx].store_action_info(action_info)
                # store oppo hidden prob
                temp_memory[1 - agent_idx].store_oppo_hidden_prob(action_info[5])
                # record
                hidden_state[agent_idx] = action_info[6]
                actions[agent_idx] = action_info[0].item()

            # env interact
            state_, reward, done, info = env.step(actions)
            # store env info
            for i in range(len(agents)):
                temp_memory[i].store_env_info(state[i], reward[i])

            # store oppo action
            if not is_prophetic:
                for agent_idx, agent in enumerate(agents):
                    if hasattr(agent, "observe_oppo_action"):
                        agent.observe_oppo_action(state=state[agent_idx], oppo_action=actions[1-agent_idx],
                                                  iteration=global_step, no_log=False)
            state = state_
            if done:
                for i in range(len(agents)):
                    temp_memory[i].store_final_state(state_[i], info)
                    memories[i].append(temp_memory[i])
                    scores[i].append(temp_memory[i].get_score())
                break
    scores = [sum(scores[i])/len(scores[i]) for i in range(len(agents))]
    return memories, scores, global_step

def collect_trajectory_for_rnn_mixer(agents, env, args, global_step, is_prophetic=False, greedy=False):
    """Collect a epoch trajectory
    :param agents: list, [agent0, agent1]
    :param env:
    :param args: contains eps_per_epoch
    :return memories: list, [2, eps_per_epoch] contains Memory
            scores: list, [2], return epoch ave reward
    """
    memories = [[], []]
    scores = [[], []]

    for _ in range(1, args.eps_per_epoch + 1):
        # Initialize LSTM state
        hidden_state = [agent.init_hidden_state(n_batch=1) for agent in agents]
        oppo_hidden_prob = np.array([None, None])
        # Begin to collect trajectory
        state = env.reset()
        temp_memory = [Episode_Memory(), Episode_Memory()]

        """ rnn_mixer train data"""
        logp_a_l = []
        value_l = []
        reward_l = []

        while True:
            global_step += 1
            actions = np.array([0, 0], dtype=int)
            for agent_idx, agent in enumerate(agents):
                if type(agent).__name__ == "MBAM":
                    '''
                     action, np.ndarray int32 (num_trj, 1, 1)
                     logp_a, np.ndarray float (num_trj, 1, 1)
                     entropy, np.ndarray float (num_trj, 1, 1)
                     value, np.ndarray float (num_trj, 1, 1)
                     action_prob, np.ndarray float (num_trj, 1, n_action)
                     hidden_prob, np.ndarray float (num_trj, 1, n_hidden_prob), this is actor network number of latest layer'cell
                     hidden_state, np.ndarray float (num_trj, 1, n_hidden_state), is None if not actor_rnn
                    '''
                    if args.rnn_mixer and agent_idx == 1:
                        action_info = agent.single_choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1 - agent_idx] if is_prophetic == True else None, greedy=greedy)
                    else:
                        action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1-agent_idx] if is_prophetic == True else None, greedy=greedy)
                elif type(agent).__name__ == "PPO":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=None, greedy=greedy)
                elif type(agent).__name__ == "Meta_mapg":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx])
                    # store oppo_log_a
                    temp_memory[agent_idx].store_oppo_logp_a(action_info[7])
                else:
                    raise TypeError

                if args.prophetic_onehot:
                    temp_a = action_info[0].item()
                    oppo_hidden_prob[agent_idx] = np.eye(action_info[5].shape[1])[temp_a].reshape(1, -1)
                    action_info = (action_info[0], action_info[1], action_info[2], action_info[3], action_info[4], oppo_hidden_prob[agent_idx], action_info[6])
                else:
                    oppo_hidden_prob[agent_idx] = action_info[5]
                # store action info
                temp_memory[agent_idx].store_action_info(action_info)
                # store oppo hidden prob
                if args.rnn_mixer and agent_idx == 1 and not args.prophetic_onehot:
                    temp_memory[1 - agent_idx].store_oppo_hidden_prob(action_info[5].detach().cpu().numpy())
                else:
                    temp_memory[1 - agent_idx].store_oppo_hidden_prob(action_info[5])
                # record
                hidden_state[agent_idx] = action_info[6]
                actions[agent_idx] = action_info[0].item()
                if agent_idx == 1:
                    """ rnn_mixer train data collect"""
                    logp_a_l.append(action_info[1])
                    value_l.append(action_info[3])
            #env.render()
            # env interact
            state_, reward, done, info = env.step(actions)
            #print(reward)
            #print("actions is " ,actions)
            # store env info
            for i in range(len(agents)):
                temp_memory[i].store_env_info(state[i], reward[i])

            # store oppo action
            if not is_prophetic:
                for agent_idx, agent in enumerate(agents):
                    if hasattr(agent, "observe_oppo_action"):
                        agent.observe_oppo_action(state=state[agent_idx], oppo_action=actions[1-agent_idx],
                                                  iteration=global_step, no_log=False)

            state = state_

            """ rnn_mixer train data collect"""
            reward_l.append(reward[1])
            """done"""
            if done:
                last_value = agents[1].v_net(torch.from_numpy(state_[1]).to(agents[1].device).float())
                agents[1].single_learn(data={"logp_a": logp_a_l,
                                             "value": value_l,
                                             "reward": reward_l,
                                             "last_val": last_value}, iteration=global_step, no_log=False)
                for i in range(len(agents)):
                    temp_memory[i].store_final_state(state_[i], info)
                    memories[i].append(temp_memory[i])
                    scores[i].append(temp_memory[i].get_score())
                break
    scores = [sum(scores[i])/len(scores[i]) for i in range(len(agents))]
    return memories, scores, global_step

if __name__ == "__main__":

    from policy.MBAM import MBAM
    from baselines.PPO import PPO, PPO_Buffer
    from env_wapper.football_penalty_kick.football_1_vs_1_penalty_kick import make_env
    from utils.Logger import Logger
    import time
    import argparse
    import numpy as np
    import torch
    import random
    starttime = time.time()

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--num_trj", type=int, default=3, help="Number of trajectories")
    parser.add_argument("--num_om_layers", type=int, default=5, help="Number of trajectories")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--actor_rnn", type=bool, default=True, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=1, help="")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="")
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
        "mix_factor": 0.1,  # adjust mix cuvre

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
        "buffer_memory_size": 3000,  # update_episode * max_episode_step = 20 * 30
        "update_episode": 50,
    }


    def env_model(state, actions, reward_idx=1):
        shape = state.shape
        state_ = torch.Tensor(np.random.random(shape)).to(device=args.device)
        reward = torch.Tensor(np.random.random((shape[0], 1))).to(device=args.device)
        done = torch.Tensor([False]).view(1, 1).repeat(shape[0], 1).to(device=args.device)
        return state_, reward, done
    logger = Logger(".", "2v2", random.randint(0, 10000))
    #mbam = MBAM(args=args, conf=conf, name="football", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
    mbam = MBAM.load_model("/home/lenovo/文档/CodeWorkspace/RL/DRL_MARL_homework/MBAM/utils/2v2/1760/model/MBAM_football_iter1000.ckp", logger, device=args.device, env_model=env_model)

    ppo = PPO(args, conf, name="football", logger=logger, actor_rnn=args.actor_rnn, device=args.device)

    env = make_env()
    global_step = 0
    agents = [ppo, mbam]

    for epoch in range(1, 1000+1):
        print("epoch:{} start!".format(epoch))
        memory, scores, global_step = collect_trajectory(agents, env, args, global_step, is_prophetic=True)

        for i in range(2):
            logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i])
            buffers[i].store_multi_memory(memory[i], last_val=0)
            agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
    print("end")