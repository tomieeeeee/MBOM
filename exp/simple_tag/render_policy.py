from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import sys
sys.path.append("D:/document/3v3")
from policy.MBAM import MBAM
from policy.MBAM_OM_MH import MBAM_OM_MH
from policy.MBAM_MH import MBAM_MH
from baselines.PPO import PPO, PPO_Buffer
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
from baselines.PPO_OM_MH import PPO_OM_MH, PPO_OM_MH_Buffer
from env_wapper.simple_tag.simple_tag import Simple_Tag
import argparse
import time
if __name__ == '__main__':
  #graphviz = GraphvizOutput()
  #graphviz.output_file = 'render.png'
  #with PyCallGraph(output=graphviz):
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--num_trj", type=int, default=3, help="Number of trajectories")
    parser.add_argument("--num_om_layers", type=int, default=5, help="Number of trajectories")
    parser.add_argument("--device", type=str, default="cpu", help="")
    parser.add_argument("--actor_rnn", type=bool, default=False, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=1, help="")
    parser.add_argument("--save_per_epoch", type=int, default=5, help="")
    parser.add_argument("--true_prob", type=bool, default=True, help="True or False, edit Actor_RNN.py line 47-48")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--only_use_last_layer_IOP", type=bool, default=False, help="True or False")
    parser.add_argument("--random_best_response", type=bool, default=False, help="True or False")
    parser.add_argument("--record_more", type=bool, default=False, help="True or False")
    parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")
    args = parser.parse_args()

    MOD = "ppo vs mbam" # or "mbam vs ppo"
    ######MOD = "mbam vs ppo"
    if MOD == "ppo vs mbam":
        #file_dir = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Tag/train/trueprob_simple_tag_ppo_vs_mbam_10oppo/0_1284741196/worker/0_2/model/"
        file_dir = "D:/document/MBAM/data/Simple_Tag/"
        player1_file = file_dir + "PPO_MH_player1_rank0_iter20000.ckp"
        player2_file = file_dir + "MBAM_player2_iter20000.ckp"
        
        player1_type = "ppo_mh" # "mbam_mh_om_mh"
        player2_type = "mbam_om_mh"

        player1_ctor = None
        player2_ctor = None
        agent1 = PPO_MH.load_model(filepath=player1_file, args=args, logger=None, device=args.device)
        agent2 = MBAM_OM_MH.load_model(filepath=player2_file, args=args, logger=None, device=args.device, env_model=None)
        env = Simple_Tag()
        agents = [agent1, agent2]
        for i in range(100):
            s = env.reset()
            while True:
                time.sleep(0.2)
                env.render()
                action_info1 = agent1.choose_action(state=s[0])
                oppo_a = [a.item() for a in action_info1[0]]
                #源代码a = agent2.choose_action(state=s[1], oppo_hidden_prob=action_info1[5])[0].item()
                action_info2 = agent2.choose_action(state=s[1], oppo_hidden_prob=action_info1[5])
                a = [a.item() for a in action_info2[0]]
                actions = [oppo_a, a]
                s_, rew, done, _ = env.step(actions)
                if rew[0] >= 10:
                    print("touch!!!!!!!")
                s = s_
                if done:
                    break
    elif MOD == "mbam vs ppo":
        #file_dir = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Tag/train/trueprob_simple_tag_mbam_vs_ppo/0_11557236/model/"
        #file_dir = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Tag/train/trueprob_simple_tag_mbam_vs_ppo/5_586138494/model/"
        #########file_dir = "D:/document/MBAM/data/test/"
        ##########player1_file = file_dir + "MBAM_player1_iter30000.ckp"
        #########player2_file = file_dir + "PPO_player2_iter30000.ckp"
        file_dir = "D:/document/MBAM/data/Simple_Tag/train/trueprob_simple_tag_ppo_vs_mbam_test/1_176448937/model/"
        player1_file = file_dir + "MBAM_player1_iter200.ckp"
        player2_file = file_dir + "PPO_player2_iter200.ckp"
        player1_type = "mbam_mh"  # "mbam_mh_om_mh"
        player2_type = "ppo"

        player1_ctor = None
        player2_ctor = None
        #load_model是静态方法staticmethod，所以不用先初始化MBAM_MH
        agent1 = MBAM_MH.load_model(filepath=player1_file, args=args, logger=None, device=args.device, env_model=None)
        agent2 = PPO.load_model(filepath=player2_file, args=args, logger=None, device=args.device)
        env = Simple_Tag()
        agents = [agent1, agent2]
        for i in range(100):
            s = env.reset()
            #s[0]和s[1]是一样的？？？？
            while True:
                time.sleep(0.2)
                env.render()
                action_info1 = agent1.choose_action(state=s[0])
                #action_info1[0]为tensor构成的列表的action：[array([2]), array([3]), array([3])]
                #a.item()指取出tensor中的方向标量，下列代码生成一个新列表
                oppo_a = [a.item() for a in action_info1[0]]
                a = agent2.choose_action(state=s[1])[0].item()
                actions = [oppo_a,a]
                s_, rew, done, _ = env.step(actions)
                if rew[0] >= 10:
                    print("touch!!!!!!!")
                s = s_
                if done:
                    break
    else:
        raise TypeError
