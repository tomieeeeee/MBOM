import sys
sys.path.append("D:/document/3v3")
import torch
import torch.nn as nn
import numpy as np
from env_wapper.simple_tag.simple_tag import Simple_Tag
from utils.datatype_transform import dcn
from utils.get_exp_data_path import get_exp_data_path
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from policy.MBAM_OM_MH import MBAM_OM_MH
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
n_agent = 6
n_state = n_agent*8+4#n*4+4
n_oppo_action = []
n_action =  []
for i in range(n_agent):
    n_oppo_action.append(5)
    n_action.append(5)
hidden_layer1 = 128
hidden_layer2 = 64
path = get_exp_data_path() + "/Simple_Tag/for_env_model/PPO_v10_1"
reward_normal_factor = 10
# 24+11+11->24+1
class ENV_Simple_Tag(nn.Module):
    def __init__(self, args):
        super(ENV_Simple_Tag, self).__init__()
        self.l1 = nn.Linear(n_state + sum(n_action) + sum(n_oppo_action), hidden_layer1)
        self.l2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.s_r = nn.Linear(hidden_layer2, n_state + 2)
        self.device = args.device

    def forward(self, x):
        x = nn.functional.tanh(self.l1(x))
        x = nn.functional.tanh(self.l2(x))
        x = nn.functional.tanh(self.s_r(x))
        return x

    def reset(self,):
        self.to(self.device)
        self.last_done=None

    def step(self, s, actions):
        '''
        :param s: [batchsize, n_state]
        :param actions: [[batchsize, oppo_action_dim] , batchsize]
        :return:
        '''
        with torch.no_grad():
            if type(s) is np.ndarray:
                s = torch.Tensor(s).view((-1, n_state)).float().to(device=self.device)
            batchsize = s.shape[0]

            oppo_a = actions[0]
            if type(oppo_a[0]) is np.ndarray:
                oppo_a = [torch.LongTensor(oppo_a[i]).view(-1).to(device=self.device) for i in range(batchsize)]
            else:
                oppo_a = [oppo_a[i].view(-1).to(device=self.device) for i in range(batchsize)]

            a = actions[1]
            if type(a) is np.ndarray:
                a = torch.LongTensor(a).view(-1).to(device=self.device)
            else:
                a = a.view(-1).to(device=self.device)


            a_oppo_onehot = [nn.functional.one_hot(oppo_a[i], num_classes=5).float().view((-1, 5)).to(device=self.device) for i in range(batchsize)]
            a_oppo_onehot = torch.stack([a_oppo_onehot[i].flatten() for i in range(batchsize)])
            #a_onehot = nn.functional.one_hot(a, num_classes=n_action).float().view((-1, n_action)).to(device=self.device)
            a_onehot = [nn.functional.one_hot(n_action[i], num_classes=5).float().view((-1, 5)).to(device=self.device) for i in range(batchsize)]
            a_onehot = torch.stack([a_oppo_onehot[i].flatten() for i in range(batchsize)])
            x = torch.cat([s, a_oppo_onehot, a_onehot], dim=1)

            x = self.forward(x)
            s_ = x[:, :n_state]
            rew = x[:, n_state:]
            #reward = torch.zeros(size=rew.shape, dtype=int, device=self.device)
            #reward[rew > 0.5] = 1
            #reward[rew < -0.5] = -1
            #done = (reward != 0).to(device=self.device)
            #if self.last_done != None:
            #    done = done | self.last_done
            #self.last_done = done
            #self.last_done = self.last_done.repeat(n_oppo_action, 1)
            #reward = torch.stack([reward, -reward], dim=0)
            reward = rew.transpose(1, 0) * reward_normal_factor
            done = torch.Tensor([False] * batchsize).bool().to(device=self.device)
            return s_, reward, done

def train(args):
    MAX_TRAIN = 100000000
    SAVE_PER_LEARN = 10000
    buf_max_size = 100000
    batch_size = 512
    cur_idx = 0
    cur_size = 0

    buf_s = np.zeros((buf_max_size, n_state), dtype=np.float32)
    #buf_a = np.zeros((buf_max_size, n_action), dtype=np.int32)
    buf_a = np.zeros((buf_max_size, sum(n_action)), dtype=np.int32)
    buf_oppo_a = np.zeros((buf_max_size, sum(n_oppo_action)), dtype=np.int32)
    buf_s_ = np.zeros((buf_max_size, n_state), dtype=np.float32)
    buf_r = np.zeros((buf_max_size, 2), dtype=np.float32)
    env = Simple_Tag()
    env_model = ENV_Simple_Tag(args).to(device=args.device)
    #env_model = load_env_model()
    optimizer = torch.optim.SGD(env_model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(path + "/log")
    agent1_dir = get_exp_data_path() + "/Simple_Tag/for_env_model/PPO_v10_1/Player1"
    agent2_dir = get_exp_data_path() + "/Simple_Tag/for_env_model/PPO_v10_1/Player2"
    agent1_paths = []
    for root, dirs, files in os.walk(agent1_dir):
        for f in files:
            agent1_paths.append(os.path.join(root, f))
    agent2_paths = []
    for root, dirs, files in os.walk(agent2_dir):
        for f in files:
            agent2_paths.append(os.path.join(root, f))
    for i in range(1, MAX_TRAIN+1):
        #if np.random.random() > 0.2:
        #gen_s, gen_a, gen_oppo_a, gen_s_, gen_r = gen_mem_with_policy(args, env, agent1_paths, agent2_paths)
        #else:
        gen_s, gen_a, gen_oppo_a, gen_s_, gen_r = gen_mem(env)
        for j in range(len(gen_s)):
            #print("buf_s",buf_s)
            #print("gen_s",gen_s)
            buf_s[cur_idx] = gen_s[j]
            buf_a[cur_idx] = gen_a[j]
            buf_oppo_a[cur_idx] = gen_oppo_a[j]
            buf_s_[cur_idx] = gen_s_[j]
            buf_r[cur_idx] = gen_r[j]
            cur_idx = (cur_idx + 1) % buf_max_size
            cur_size = min(cur_size + 1, buf_max_size)
        #print("buffer cur_idx:{}".format(cur_idx))
        if cur_size < buf_max_size:
            print("buffer isn't full! {}/{}".format(cur_size, buf_max_size))
            continue
        sample_idx = np.random.choice([i for i in range(buf_max_size)], batch_size)
        batch_s = buf_s[sample_idx]
        batch_oppo_a = buf_oppo_a[sample_idx]
        batch_a = buf_a[sample_idx]
        batch_s_ = buf_s_[sample_idx]
        batch_r = buf_r[sample_idx]
        x = np.hstack([batch_s, batch_oppo_a, batch_a])
        target = np.hstack([batch_s_, batch_r.reshape(-1, 2)])
        target = torch.Tensor(target).to(device=args.device)
        optimizer.zero_grad()
        if type(x) is np.ndarray:
            x = torch.Tensor(x).to(device=args.device)
        eval = env_model(x)
        loss = loss_fn(eval, target)
        print("loss: %f" % float(loss))
        writer.add_scalar("loss", float(loss), i)
        loss.backward()
        optimizer.step()
        if i % SAVE_PER_LEARN == 0:
            torch.save(env_model, path + "/%i.pt" % i)
    writer.close()

def gen_mem(env):
    #####!!!!!!!!!!!!!!!!!!!!!!  先opponent_action 再 action!!!!!!!!!
    def random_actions():
        return np.random.randint(0, n_action, size=[2])
    gen_s = []
    gen_a = []
    gen_oppo_a = []
    gen_s_ = []
    gen_r = []
    s = env.reset()
    s = s[0]
    while True:
        #actions = random_actions()
        #actions = [[np.random.randint(0, 5, 3) for i in range(1)],np.random.randint(0, 5, 1)]
        
        list_oppo = []
        list_a = []

       #list = [[ np.random.randint(0, 5, size=[1]),np.random.randint(0, 5, size=[1]),np.random.randint(0, 5, size=[1])],[ np.random.randint(0, 5, size=[1]),np.random.randint(0, 5, size=[1]),np.random.randint(0, 5, size=[1])]]
        for i in range(n_agent):
            list_oppo.append(np.random.randint(0, 5, size=[1]))
            list_a.append(np.random.randint(0, 5, size=[1]))
        #生成形如[[array([3]), array([1]), array([2])], array([4])]的列表

        list = [list_oppo,list_a]
        actions = [[action.item() for  action in list[0]],[action.item() for  action in list[1]]]
        #print("actions",actions)
        s_, r, d, _ = env.step(actions)
        '''
        s_ = s_[1]
        gen_s.append(s)
        gen_a.append(np.eye(n_action)[actions[0]])
        gen_oppo_a.append(np.eye(n_oppo_action)[actions[1]])
        gen_s_.append(s_)
        gen_r.append(r[1])
        s = s_
        '''
        s_ = s_[1]
        gen_s.append(s.copy())
        gen_oppo_a.append(np.concatenate([np.eye(n_oppo_action[i])[actions[0][i]] for i in range(len(actions[0]))]))
        #gen_a.append(np.eye(n_action)[actions[1]])
        gen_a.append(np.concatenate([np.eye(n_action[i])[actions[1][i]] for i in range(len(actions[1]))]))
        gen_r.append(np.array(r)/reward_normal_factor)
        gen_s_.append(s_)
        '''
        print(np.concatenate([np.eye(n_oppo_action[i])[actions[0][i]] for i in range(len(actions[0]))]))
        print(type(np.concatenate([np.eye(n_oppo_action[i])[actions[0][i]] for i in range(len(actions[0]))])))
        print(np.eye(n_action)[actions[1]])
        print(type(np.eye(n_action)[actions[1]]))
        print(np.array(r)/reward_normal_factor)
        print(type(np.array(r)/reward_normal_factor))
        print(r)
        print(type(r))
        print(np.array(r))
        print(type(np.array(r)))
        print(np.array(r))
        '''
        if d:
            break
    return gen_s, gen_a, gen_oppo_a, gen_s_, gen_r
    pass

def gen_mem_with_policy(args, env, agent1_paths, agent2_paths):
    gen_s = []
    gen_a = []
    gen_oppo_a = []
    gen_s_ = []
    gen_r = []
    path1 = agent1_paths[np.random.randint(0, len(agent1_paths), 1).item()]
    path2 = agent2_paths[np.random.randint(0, len(agent2_paths), 1).item()]

    agent1 = PPO_MH.load_model(filepath=path1, args=args, logger=None, device=args.device)
    agent2 = MBAM_OM_MH.load_model(filepath=path2, args=args, logger=None, device=args.device, env_model=None)

    obs = env.reset()
    hidden_state1 = agent1.init_hidden_state(n_batch=1)
    hidden_state2 = agent2.init_hidden_state(n_batch=1)
    while True:
        action_info1 = agent1.choose_action(obs[0], hidden_state=hidden_state1)
        action_info2 = agent2.choose_action(obs[1], hidden_state=hidden_state2, oppo_hidden_prob=action_info1[5])
        #actions = random_actions()
        #print(type(action_info1))
        #print(action_info1)
        actions = [[a.item() for a in action_info1[0]], action_info2[0].item()]
        obs_, r, d, _ = env.step(actions)
        gen_s.append(obs[0].copy())
        gen_oppo_a.append(np.concatenate([np.eye(n_oppo_action[i])[actions[0][i]] for i in range(len(actions[0]))]))
        gen_a.append(np.eye(n_action)[actions[1]])
        gen_s_.append(obs_[0].copy())
        gen_r.append(np.array(r)/reward_normal_factor)
        obs = obs_
        if d:
            break
    return gen_s, gen_a, gen_oppo_a, gen_s_, gen_r
    pass


def load_env_model(device):
    if not os.path.exists(path + "/850000.pt"):
        print("Env model don't exist!", path + "/850000.pt")
    try:
        env_model = torch.load(path + "/850000.pt", map_location='cpu')
    except Exception as e:
        print("error:", e)
    env_model.device = device
    return env_model
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PyTorch RL model Training')
    # parser.add_argument('--train', default=False, type=bool, help='')
    # parser.add_argument('--device', default="cuda", type=str, help='')
    # __main__(parser.parse_args())


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
    parser.add_argument("--only_use_last_layer_IOP", type=bool, default=False, help="True or False")

    parser.add_argument('--train', default=False, type=bool, help='')
    args = parser.parse_args()
    #__main__(parser.parse_args())
    train(args)
    env = load_env_model(device='cpu')
    pass
    # test code

    #env = ENV_Simple_Tag(args)
    batchsize = 9
    state = np.random.random((batchsize, 20))

    actions = [[np.random.randint(0, 5, 3) for i in range(batchsize)], np.random.randint(0, 5, batchsize)]
    result = env.step(state, actions)
    print(result)
    pass
    ''''''