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


    #file_dir = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Tag/train/trueprob_simple_tag_ppo_vs_mbam_10oppo/0_1284741196/worker/0_2/model/"
    file_dir = "D:/document/MBAM/data/for_data/PPO_v10_1/set1/4_1364118395/worker/0_0/model/"
    player1_file = file_dir + "PPO_MH_player1__player1_iter44900.ckp"
    player2_file = file_dir + "PPO_MH_player2_player2_iter44900.ckp"
    #PPO_MH_player1__player1_iter120700
    player1_type = "ppo_mh" # "mbam_mh_om_mh"
    player2_type = "mbam_om_mh"

    

    player1_ctor = None
    player2_ctor = None
    agent1 = PPO_MH.load_model(filepath=player1_file, args=args, logger=None, device=args.device)
    agent2 = PPO_MH.load_model(filepath=player2_file, args=args, logger=None, device=args.device)#, env_model=None
    env = Simple_Tag()
    agents = [agent1, agent2]
    score=[]
    score1=[]

    for i in range(100):
        dis_oppo = 0
        dis_me = 0
        temp=[]
        temp1=[]
        step = 0
        s = env.reset()
        while True:
            #env.render()
            #env.render("mode=rgb_array")
            #print(type(frame))
            action_info1 = []
            
            start = time.time()


            for i in range (10):
                action_info1.append(agent1.choose_action(state=s[i][0]))
            oppo_a = []
            for i in range (10):
                oppo_a.append([a.item() for a in action_info1[i][0]])
            end = time.time()
            #源代码a = agent2.choose_action(state=s[1], oppo_hidden_prob=action_info1[5])[0].item()
            start = time.clock()
            action_info2 = []
            
            for i in range (10):
                action_info2.append(agent2.choose_action(state=s[i][1]))
            end = time.clock()

            a = []

            for i in range (10):
                a.append([a.item() for a in action_info2[i][0]])   
            oppo_action = []
            
            actions = [oppo_a,a]
            for i in range(len(oppo_a)):
                for num in oppo_a[i]:
                    if num != 0 :
                        dis_oppo += 1
            for i in range(len(a)):
                for num in a[i]:
                    if num != 0 :
                        dis_me += 1
            #print("步数",step)
            s_, rew, done, _ = env.step(actions)
            temp.append(rew[0])
            temp1.append(rew[1])
            step+=1
    #print(rew)
    #if rew[0] >= 10:
    #    print("touch!!!!!!!")
            s = s_
            if done:
                break
 
