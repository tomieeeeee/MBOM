import sys
sys.path.append('/home/lenovo/文档/CodeWorkspace/RL')
import torch
import torch.nn as nn
import numpy as np
from DRL_MARL_homework.MBAM.env_wapper.mpe.simple_keepaway import Simple_Keep_away
from DRL_MARL_homework.MBAM.utils.datatype_transform import dcn
from DRL_MARL_homework.MBAM.utils.get_exp_data_path import get_exp_data_path
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

'''
state 修改,将2个agent的obs合并,dim=27
action修改, 原始action为 上、下、左、右、空 5个方向进行的。 现在将
为0-18,改为:进攻方(left)使用[0-8,12,14]N个;防守方(right)使用[0-8,9,16]N个
env = gfootball.env.create_environment(env_name="tests.1_vs_1_penalty_kick",
                                           representation="simple115",
                                           rewards='scoring, checkpoints',  # 'scoring',
                                           number_of_left_players_agent_controls= 1,
                                           number_of_right_players_agent_controls=1,
                                           # stacked=False,
                                           # logdir='./log/',
                                           # write_goal_dumps=True,
                                           # write_full_episode_dumps=False,
                                           # write_video=False,
                                           # dump_frequency=1000,
                                           )
'''
n_state = 27
n_action = 5
n_oppo_action = 5
hidden_layer1 = 64
hidden_layer2 = 32
path = get_exp_data_path() + "/Simple_Keep_away/Env_Model/ENV_keepaway"
# 24+11+11->24+1
class ENV_Simple_Keep_away(nn.Module):
    def __init__(self, args):
        super(ENV_Simple_Keep_away, self).__init__()
        self.l1 = nn.Linear(n_state + n_action + n_oppo_action, hidden_layer1)
        self.l2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.s_r = nn.Linear(hidden_layer2, n_state + 1)
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
        with torch.no_grad():
            if type(s) is np.ndarray:
                s = torch.Tensor(s).view((-1, n_state)).float().to(device=self.device)
            a = actions[0]
            if type(a) is np.ndarray:
                a = torch.LongTensor(a).view(-1).to(device=self.device)
            else:
                a = a.view(-1).to(device=self.device)
            oppo_a = actions[1]
            if type(oppo_a) is np.ndarray:
                oppo_a = torch.LongTensor(oppo_a).view(-1).to(device=self.device)
            else:
                oppo_a = oppo_a.view(-1).to(device=self.device)
            a_onehot = nn.functional.one_hot(a, num_classes=n_action).float().view((-1, n_action)).to(device=self.device)
            a_oppo_onehot = nn.functional.one_hot(oppo_a, num_classes=n_oppo_action).float().view((-1, n_oppo_action)).to(device=self.device)
            x = torch.cat([s, a_onehot, a_oppo_onehot], dim=1)

            x = self.forward(x)
            s_ = x[:, :n_state]
            rew = x[:, n_state:]
            reward = torch.zeros(size=rew.shape, dtype=int, device=self.device)
            reward[rew > 0.5] = 1
            reward[rew < -0.5] = -1
            done = (reward != 0).to(device=self.device)
            if self.last_done != None:
                done = done | self.last_done
            self.last_done = done
            self.last_done = self.last_done.repeat(n_oppo_action, 1)
            reward = torch.stack([reward, -reward], dim=0)
            return s_, reward, done

def train(args):
    MAX_TRAIN = 100000000
    SAVE_PER_LEARN = 10000
    buf_max_size = 100000
    batch_size = 512
    cur_idx = 0
    cur_size = 0

    buf_s = np.zeros((buf_max_size, n_state), dtype=np.float32)
    buf_a = np.zeros((buf_max_size, n_action), dtype=np.int32)
    buf_oppo_a = np.zeros((buf_max_size, n_oppo_action), dtype=np.int32)
    buf_s_ = np.zeros((buf_max_size, n_state), dtype=np.float32)
    buf_r = np.zeros(buf_max_size, dtype=np.float32)
    env = Simple_Keep_away()
    env_model = ENV_Simple_Keep_away(args).to(device=args.device)
    #env_model = load_env_model()
    optimizer = torch.optim.SGD(env_model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(path + "/log")
    for i in range(1, MAX_TRAIN+1):
        gen_s, gen_a, gen_oppo_a, gen_s_, gen_r = gen_mem(env)
        for j in range(len(gen_s)):
            buf_s[cur_idx] = gen_s[j]
            buf_a[cur_idx] = gen_a[j]
            buf_oppo_a[cur_idx] = gen_oppo_a[j]
            buf_s_[cur_idx] = gen_s_[j]
            buf_r[cur_idx] = gen_r[j]
            cur_idx = (cur_idx + 1) % buf_max_size
            cur_size = min(cur_size + 1, buf_max_size)
        if cur_size < buf_max_size:
            print("buffer isn't full! {}/{}".format(cur_size, buf_max_size))
            continue
        sample_idx = np.random.choice([i for i in range(buf_max_size)], batch_size)
        batch_s = buf_s[sample_idx]
        batch_a = buf_a[sample_idx]
        batch_oppo_a = buf_oppo_a[sample_idx]
        batch_s_ = buf_s_[sample_idx]
        batch_r = buf_r[sample_idx]
        x = np.hstack([batch_s, batch_a, batch_oppo_a])
        target = np.hstack([batch_s_, batch_r.reshape(-1, 1)])
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
        actions = random_actions()
        s_, r, d, _ = env.step(actions)
        s_ = s_[0]
        gen_s.append(s)
        gen_a.append(np.eye(n_action)[actions[0]])
        gen_oppo_a.append(np.eye(n_oppo_action)[actions[1]])
        gen_s_.append(s_)
        gen_r.append(r[0])
        s = s_
        if d:
            break
    return gen_s, gen_a, gen_oppo_a, gen_s_, gen_r
    pass

def load_env_model(device):
    if not os.path.exists(path + "/2750000.pt"):
        print("Env model don't exist!", path + "/2750000.pt")
    env_model = torch.load(path + "/2750000.pt", map_location='cpu')
    env_model.device = device
    return env_model

def __main__(args):
    if args.train:
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch RL model Training')
    parser.add_argument('--train', default=False, type=bool, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    __main__(parser.parse_args())
    pass