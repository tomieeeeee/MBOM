from env_wapper.mpe.make_env import make_env
import numpy as np


class Simple_RPS(object):
    def __init__(self, eps_max_step=100, version="simple_rps"):
        super(Simple_RPS, self).__init__()
        self.n_agent = 2
        self.n_state = 14  #以自己为中心，超出4格以外则结束  [0, 1, 2, 3 ,4 ,5, 6, 7, 8]
        self.n_action = 11
        self.n_opponent_action = 11

        self.actions_trans = np.array([[1, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                       [0, 0.5, 0, 0.8660, 0], [0, 0, 0.5, 0.8660, 0],
                                       [0, 0.5, 0, 0, 0.8660], [0, 0, 0.5, 0, 0.8660],
                                       [0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5]])
        self.env_state_running = False
        self.eps_max_step = eps_max_step
        self.cur_step = 0
        if version == 'simple_rps':
            self.env = make_env('simple_rps')
        elif version == 'simple_rps_v1':
            self.env = make_env('simple_rps_v1')
        elif version == 'simple_rps_v2':
            self.env = make_env('simple_rps_v2')
        else:
            raise NameError

    def reset(self):
        self.cur_step = 0
        self.env_state_running = True
        return self.env.reset()

    def step(self, actions):
        self.cur_step += 1
        assert self.env_state_running == True, "Env is stoped, please reset()"
        a = self.actions_trans[actions[0]]
        oppo_a = self.actions_trans[actions[1]]
        obs_, rew, _, info = self.env.step([a, oppo_a])
        done = False
        if self.cur_step == self.eps_max_step:
            done = True
            self.env_state_running = False
        for i in range(2):
            if (np.abs(obs_[i][1:]) >= obs_[i][0]).any() == True:
                done = True
                rew[i] = -self.eps_max_step
        return obs_, rew, done, info

    def render(self):
        self.env.render()

if __name__ == "__main__":
    from policy.MBAM import MBAM
    from baselines.PPO import PPO
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
    parser.add_argument("--true_prob", type=bool, default=True, help="True or False")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--policy_training", type=bool, default=False, help="True or False")
    args = parser.parse_args()


    #path1 = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam/1_1188284716/worker/2_2/model/PPO_player1_rank2_iter41400.ckp"
    #path2 = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam/1_1188284716/model/MBAM_trigame2_iter41400.ckp"

    """simple_rps_v1"""
    #dir = ["2_0", "1_1", "0_2", "4_3", "5_4", "3_5", "6_6", "7_7", "8_8", "9_9"]     # 6 2 2 上左右
    #iter = 13600
    #path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v1/2_492096035/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    #path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v1/2_492096035/worker/{}/model/MBAM_player2_iter{}.ckp"
    """simple_rps_v2"""
    #dir = ["0_0", "1_1", "4_2", "2_3", "3_4", "6_5", "5_6", "9_7", "8_8", "7_9"]     #4 6
    #iter = 14400
    #path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v2/0_427309454/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    #path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v2/0_427309454/worker/{}/model/MBAM_player2_iter{}.ckp"
    """simple_rps_v2"""
    #dir = ["1_0", "3_1", "7_2", "2_3", "6_4", "4_5", "9_6", "0_7", "8_8", "5_9"]  #  3 4
    #iter = 24300
    #path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v2/1_851265259/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    #path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v2/1_851265259/worker/{}/model/MBAM_player2_iter{}.ckp"
    """simple_rps_v0"""
    #dir = ["1_0", "4_1", "2_2", "3_3", "7_4", "0_5", "8_6", "9_7", "5_8", "6_9"]  # 2 9
    #iter = 10000
    #path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v0/2_151954998/worker/{}/model/PPO_player1_rank{}_iter{}.ckp"
    #path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_ppo_vs_mbam_10oppo_v0/2_151954998/worker/{}/model/MBAM_player2_iter{}.ckp"

    """
    env = Simple_RPS(args.eps_max_step)
    for i in range(10):
        #i=6
        path1 = path1_T.format(dir[i], i, iter)
        path2 = path2_T.format(dir[i], iter)
        agent1 = PPO.load_model(path1, args, logger=None, device="cuda")
        agent2 = MBAM.load_model(path2, args, logger=None, device="cuda", env_model=None)

        obs = env.reset()
        hidden_state1 = agent1.init_hidden_state(n_batch=1)
        hidden_state2 = agent2.init_hidden_state(n_batch=1)
        steps = 0
        sum_reward = [0, 0]
        while True:
            steps += 1
            env.render()
            action_info1 = agent1.choose_action(obs[0], hidden_state=hidden_state1, greedy=True)
            action_info2 = agent2.choose_action(obs[1], hidden_state=hidden_state2, oppo_hidden_prob=action_info1[5], greedy=True)

            #action = np.random.randint(0, 9)
            #oppo_action = np.random.randint(0, 9)
            actions = np.hstack([action_info1[0].item(), action_info2[0].item()])
            # action = 0
            # oppo_action = 0
            #actions = [0, 1]
            obs, rew, done, _ = env.step(actions)
            #print(obs)
            sum_reward[0] += rew[0]
            sum_reward[1] += rew[1]
            if done:
                print("{}, {}, {}".format(i, steps, sum_reward))
                time.sleep(1)
                break
    """

    # """MBAM vs PPO"""  #  0_823962281 蓝不动  4_1379425723  中间乱走
    seed = ["0_1449317377", "1_1310095348", "2_150906803", "3_1284318016", "4_1526911029", "5_883396975", "6_770111481", "7_1359047015", "8_1069536043"]
    iter = 20000
    path1_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_mbam_vs_ppo/{}/model/MBAM_player1_iter{}.ckp"
    path2_T = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/train/trueprob_simple_rps_mbam_vs_ppo/{}/model/PPO_player2_iter{}.ckp"

    path11 = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/Player1/for_test_2/0_1449317377/model/MBAM_player1_iter6000.ckp"
    env = Simple_RPS(args.eps_max_step)
    for i in range(9):
        path1 = path1_T.format(seed[i], iter)
        path2 = path2_T.format(seed[i], iter)
        agent1 = MBAM.load_model(path11, args, logger=None, device="cuda", env_model=None)
        agent2 = PPO.load_model(path2, args, logger=None, device="cuda")

        obs = env.reset()
        hidden_state1 = agent1.init_hidden_state(n_batch=1)
        hidden_state2 = agent2.init_hidden_state(n_batch=1)
        steps = 0
        sum_reward = [0, 0]
        while True:
            steps += 1
            env.render()
            action_info2 = agent2.choose_action(obs[1], hidden_state=hidden_state2)
            action_info1 = agent1.choose_action(obs[0], hidden_state=hidden_state1, oppo_hidden_prob=action_info2[5])


            # action = np.random.randint(0, 9)
            # oppo_action = np.random.randint(0, 9)
            actions = np.hstack([action_info1[0].item(), action_info2[0].item()])
            # action = 0
            # oppo_action = 0
            # actions = [0, 1]
            obs, rew, done, _ = env.step(actions)
            # print(obs)
            sum_reward[0] += rew[0]
            sum_reward[1] += rew[1]
            time.sleep(0.2)
            if done:
                print("{}, {}".format(steps, sum_reward))
                break