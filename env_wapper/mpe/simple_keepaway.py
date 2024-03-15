from DRL_MARL_homework.MBAM.env_wapper.mpe.make_env import make_env
import numpy as np


class Simple_Keep_away(object):
    def __init__(self, eps_max_step=30, env_version="simple_push"):
        super(Simple_Keep_away, self).__init__()
        self.n_agent = 2
        self.n_state = 27
        self.n_action = 9
        self.n_opponent_action = 9

        self.actions_trans = np.array([[1, 0, 0, 0, 0],
                                       [0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0],
                                       [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5],
                                       [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5, 0],
                                       [0, 0.5, 0, 0, 0.5], [0, 0, 0.5, 0, 0.5],
                                       ])
        self.env_state_running = False
        self.eps_max_step = eps_max_step
        self.cur_step = 0
        if env_version == "simple_push":
            self.env = make_env('simple_push')
        elif env_version == "simple_push_v1":
            self.env = make_env('simple_push_v1')
        elif env_version == "simple_push_v2":
            self.env = make_env('simple_push_v2')
        elif env_version == "simple_push_v3":
            self.env = make_env('simple_push_v3')
        else:
            raise NameError

    def reset(self):
        self.cur_step = 0
        self.env_state_running = True
        obs = self.env.reset()
        #obs_new = np.concatenate(obs).reshape(1, -1).repeat(2, axis=0)
        #obs_new = [obs[0], np.concatenate(obs)]
        return obs

    def step(self, actions):
        self.cur_step += 1
        assert self.env_state_running == True, "Env is stoped, please reset()"
        a = self.actions_trans[actions[0]]
        oppo_a = self.actions_trans[actions[1]]
        obs_, rew, done, info = self.env.step([a, oppo_a])
        #obs_ = np.concatenate(obs_).reshape(1, -1).repeat(2, axis=0)
        #obs_new = [obs_[0], np.concatenate(obs_)]
        done = np.any(done)
        if self.cur_step == self.eps_max_step:
            done = True
            self.env_state_running = False
        return obs_, rew, done, info

    def render(self):
        self.env.render()

    def set_agent2_target(self, is_close_to_goal, target_thres):
        self.env.world.is_close_to_goal = is_close_to_goal
        self.env.world.target_thres = target_thres


if __name__ == "__main__":
    from DRL_MARL_homework.MBAM.policy.MBAM import MBAM
    from DRL_MARL_homework.MBAM.baselines.PPO import PPO
    import argparse
    import time

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--exp_name", type=str, default="predator_ppo_vs_mbam", help="football_ppo_vs_mbam\n " +
                                                                                     "predator_ppo_vs_mbam\n" +
                                                                                     "rps_ppo_vs_mbam\n" +
                                                                                     "keepaway_ppo_vs_mbam\n" +
                                                                                     "trigame_ppo_vs_mbam\n" +
                                                                                     "simple_rps_ppo_vs_mbam")
    parser.add_argument("--env", type=str, default="predator", help="football\n" +
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

    parser.add_argument("--eps_max_step", type=int, default=50, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=10, help="")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="")
    parser.add_argument("--max_epoch", type=int, default=100, help="train epoch")
    parser.add_argument("--num_om_layers", type=int, default=3, help="train epoch")
    parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")

    parser.add_argument("--actor_rnn", type=bool, default=False, help="True or False")
    parser.add_argument("--true_prob", type=bool, default=False, help="True or False")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--policy_training", type=bool, default=False, help="True or False")
    args = parser.parse_args()

    # path1 = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam/1_1188284716/worker/2_2/model/PPO_player1_rank2_iter41400.ckp"
    # path2 = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam/1_1188284716/model/MBAM_trigame2_iter41400.ckp"

    """simple_keepaway_v3"""
    #dir = ["0_0", "0_1", "2_2", "5_3", "3_4", "1_5", "2_6", "6_7"]
    #dir = ["5_0", "0_1", "2_2", "4_3", "3_4", "1_5"]
    #iter = 34000
    #path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Keep_away/train/rnn_keepaway_ppo_vs_mbam_for_targeted_ppo/0_1340345499/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    #path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Keep_away/train/rnn_keepaway_ppo_vs_mbam_for_targeted_ppo/0_1340345499/worker/{}/model/MBAM_player2_iter{}.ckp"
    """simple_keepaway_v2"""
    #dir = ["0_0", "1_1", "4_2", "5_3", "3_4", "7_5", "2_6", "6_7"]     #4 6
    #iter = 70000
    #path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Keep_away/train/rnn_keepaway_ppo_vs_mbam/1_265803813/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    #path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Keep_away/train/rnn_keepaway_ppo_vs_mbam/1_265803813/worker/{}/model/MBAM_player2_iter{}.ckp"
    """simple_keepaway_v2"""
    dir = ["0_0", "5_1", "1_2", "3_3", "4_4", "2_5"]
    iter = 10000
    path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Keep_away/train/rnn_keepaway_ppo_vs_mbam_for_targeted_ppo/1_1265046028/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Keep_away/train/rnn_keepaway_ppo_vs_mbam_for_targeted_ppo/1_1265046028/worker/{}/model/MBAM_player2_iter{}.ckp"
    """simple_keepaway_v0"""
    # dir = ["1_0", "4_1", "2_2", "3_3", "7_4", "0_5", "8_6", "9_7", "5_8", "6_9"]  # 2 9
    # iter = 10000
    # path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v0/2_151954998/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    # path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v0/2_151954998/worker/{}/model/MBAM_player2_iter{}.ckp"


    env = Simple_Keep_away(args.eps_max_step, "simple_push_v3")
    for i in range(len(dir)):
        path1 = path1_T.format(dir[i], i, iter)
        path2 = path2_T.format(dir[i], iter)
        agent1 = PPO.load_model(path1, args, logger=None, device="cuda")
        #agent2 = MBAM.load_model(path2, args, logger=None, device="cuda", env_model=None)

        obs = env.reset()
        hidden_state1 = agent1.init_hidden_state(n_batch=1)
        #hidden_state2 = agent2.init_hidden_state(n_batch=1)
        steps = 0
        sum_reward = [0, 0]
        while True:
            steps += 1
            env.render()
            action_info1 = agent1.choose_action(obs[0], hidden_state=hidden_state1, greedy=True)
            #action_info2 = agent2.choose_action(obs[1], hidden_state=hidden_state2, oppo_hidden_prob=action_info1[5], greedy=True)
            action_info2 = np.random.randint(0, 9, size=1)

            #action = np.random.randint(0, 9)
            #oppo_action = np.random.randint(0, 9)
            actions = np.hstack([action_info1[0].item(), action_info2[0].item()])
            #actions = np.hstack([action_info1[0].item(), 0])
            # action = 0
            # oppo_action = 0
            #actions = [0, 1]
            obs, rew, done, _ = env.step(actions)
            #print(obs)
            sum_reward[0] += rew[0]
            sum_reward[1] += rew[1]
            time.sleep(0)
            print(rew)
            if done:
                print("{}, {}, {}".format(i, steps, sum_reward))
                time.sleep(1)
                break
