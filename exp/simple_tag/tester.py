# RL model and buffer
from policy.MBAM_OM_MH import MBAM_OM_MH
from baselines.PPO_OM_MH import PPO_OM_MH_Buffer
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
from policy.MBAM_MH import MBAM_MH
from baselines.PPO import PPO, PPO_Buffer
# env
from env_wapper.simple_tag.simple_tag import Simple_Tag
# env_model
from env_model.simple_tag.model_simple_tag import load_env_model as simple_tag_env_model
# conf
from config.simple_tag_conf import player1_conf, player2_conf
from utils.Logger import Logger
from utils.rl_utils_MH import collect_trajectory_MH
from utils.get_process_memory import get_current_memory_gb
from utils.get_exp_data_path import get_exp_data_path
import os
import random
import torch
import numpy as np
import multiprocessing as mp
# RL model and buffer
from policy.MBAM_OM_MH import MBAM_OM_MH
from baselines.PPO_OM_MH import PPO_OM_MH_Buffer
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
from policy.MBAM_MH import MBAM_MH
from baselines.PPO import PPO, PPO_Buffer
# env
from env_wapper.simple_tag.simple_tag import Simple_Tag
# env_model
from env_model.simple_tag.model_simple_tag import load_env_model as simple_tag_env_model
# conf
from config.simple_tag_conf import player1_conf, player2_conf
from utils.Logger import Logger
from utils.rl_utils_MH import collect_trajectory_MH
from utils.get_process_memory import get_current_memory_gb
from utils.get_exp_data_path import get_exp_data_path
import os
import random
import torch
import numpy as np
import multiprocessing as mp

# RL model and buffer
from policy.MBAM_OM_MH import MBAM_OM_MH
from baselines.PPO_OM_MH import PPO_OM_MH_Buffer
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
from policy.MBAM_MH import MBAM_MH
from baselines.PPO import PPO, PPO_Buffer
# env
from env_wapper.simple_tag.simple_tag import Simple_Tag
# env_model
from env_model.simple_tag.model_simple_tag import load_env_model as simple_tag_env_model
# conf
from config.simple_tag_conf import player1_conf, player2_conf
from utils.Logger import Logger
from utils.rl_utils_MH import collect_trajectory_MH
from utils.get_process_memory import get_current_memory_gb
from utils.get_exp_data_path import get_exp_data_path
import os
import random
import torch
import numpy as np
import multiprocessing as mp

def simple_tag_tester(args, logger, **kwargs):
    # '''set seed'''
    # random.seed(logger.seed)
    # torch.manual_seed(logger.seed)
    # np.random.seed(logger.seed)
    '''get shooter model file list'''
    player1_path = get_exp_data_path() + "/Simple_Tag/Player1/for_test_{}".format(args.test_mode)
    player1_model_file_list = []
    for root, dirs, files in os.walk(player1_path):
        for f in files:
            player1_model_file_list.append(os.path.join(root, f))
    player1_model_file_list.sort()
    '''get goalkeeper model file'''
    if args.player2_is_ppo:
        player2_ckp = get_exp_data_path() + "/Simple_Tag/PPO_player2_iter30000.ckp"
    else:
        player2_ckp = get_exp_data_path() + "/Simple_Tag/MBAM_player2_iter7500_run.ckp"
    
    res_l = []
    process_count = 0
    pool = mp.Pool(processes=args.test_mp)
    
    for player1_id, file in enumerate(player1_model_file_list):
        process_count += 1
        
        res = pool.apply_async(test_worker, (args, logger.root_dir, player1_id, file, player2_ckp, logger.seed, player1_conf, player2_conf))
        
        #res = test_worker(args, logger.root_dir, player1_id, file, player2_ckp, logger.seed, player1_conf, player2_conf)
        res_l.append(res)
        if process_count == args.test_mp:
            process_count = 0
            pool.close()
            pool.join()
            pool = mp.Pool(processes=args.test_mp)
    pool.close()
    pool.join()

def test_worker(args, root_dir, player1_id, player1_file, player2_ckp, seed, player1_conf=None, player2_conf=None):
    '''set seed'''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    '''set logger'''
    cur_logger = Logger(root_dir, "rank", player1_id)
    '''prepare env'''
    env = Simple_Tag(args.eps_max_step)
    #env_model = simple_tag_env_model(args.device)
    env_model = None
    logger = Logger(root_dir, "worker", 0)
    '''prepare agents'''
    if args.test_mode == 0 or args.test_mode == 1:
        
        agent1 = PPO_MH(args, player1_conf, name="player1_", logger=logger,actor_rnn=args.actor_rnn, device=args.device)
        #agent1 = PPO_MH.load_model(player1_file, args, cur_logger, args.device)
        
    elif args.test_mode == 2:
        agent1 = MBAM_MH.load_model(player1_file, args, cur_logger, args.device, env_model=None)
        agent1.agent_idx = 0
        #temp = PPO.load_model(player1_file, args, cur_logger, args.device)
        #agent1 = MBAM(args, player1_conf, name="player1{}".format(player1_id), logger=cur_logger, agent_idx=0, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
        #agent1.a_net.load_state_dict(temp.a_net.state_dict())
    else:
        raise NameError
    #agent2 = MBAM(args=args, conf=goalkeeper_conf, name="goalkeeper", logger=cur_logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
    
    if args.player2_is_ppo:
        try:
            
            agent2 = PPO_MH(args=args, conf=player2_conf, name="player2", logger=logger,actor_rnn=args.actor_rnn, device=args.device)
 
            #agent2 = PPO.load_model(player2_ckp, args, cur_logger, args.device)
            agent2.name = agent2.name + "_player2"
            buffer2 = PPO_MH_Buffer(args, player2_conf, agent2.name, actor_rnn=args.actor_rnn, device=args.device)
        except Exception as e:
            print(e)
    else:
        agent2 = MBAM_OM_MH.load_model(player2_ckp, args, cur_logger, device=args.device, env_model=env_model)
        agent2.name = agent2.name + "_player2"
        buffer2 = PPO_OM_MH_Buffer(args, player2_conf, agent2.name, actor_rnn=args.actor_rnn, device=args.device)
    agent1.name = agent1.name + "_player1"
    agents = [agent1, agent2]
    buffer1 = PPO_MH_Buffer(args, player1_conf, agent1.name, actor_rnn=args.actor_rnn, device=args.device)
    buffers = [buffer1, buffer2]
    "init opponent model param"
    #if args.test_mode == 0 or args.test_mode == 1:
    #    agent2.oppo_model.model.load_state_dict(agent1.a_net.state_dict())
    '''change param'''
    if player1_conf is not None:
        agent1.conf = player1_conf
    if args.prefix == "test":
        agent1.conf["v_learning_rate"] = 0.01
        agent1.conf["a_learning_rate"] = 0.01
        agent2.conf["v_learning_rate"] = 0.01
        agent2.conf["a_learning_rate"] = 0.01
    if player2_conf is not None:
        agent2.conf = player2_conf
    if args.prefix == "test":
        agent2.conf["mix_factor"] = 1
        agent2.conf["opponent_model_learning_times"] = 3
        agent2.conf["imagine_model_learning_rate"] = 0.01
        agent2.conf["opponent_model_learning_rate"] = 0.001
    if not args.player2_is_ppo:
        agent2.change_om_layers(args.num_om_layers, args.rnn_mixer)
    '''log param'''
    cur_logger.log_param(args, [agent.conf for agent in agents])

    '''test agent'''
    global_step = 0
    for epoch in range(1,  args.max_epoch + 1):#
        cur_logger.log("rank:{}! epoch:{} start!".format(player1_id, epoch))
        
        '''collect_trajectory'''
        try:
            memory, scores, global_step, touch_times = collect_trajectory_MH(agents, env, args, global_step, is_prophetic=True)
            
        except Exception as e:
            print(e)
        '''learn'''
        
        for i in range(2):
            cur_logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i], Touch=touch_times)
            if args.test_mode == 0 and i == 0:
                pass
            else:
                if args.policy_training == False and i == 1:
                    pass
                else:
                    
                    assert args.env == "simple_tag", "env is not simple_tag"
                    last_val = [agents[i].v_net(torch.from_numpy(m.final_state[0].astype(np.float32)).to(args.device)).detach().cpu().numpy().item() for m in memory[i]]
                    
                    buffers[i].store_multi_memory(memory[i], last_val=last_val)
                    #buffers[i].store_multi_memory(memory[i], last_val=0)
                    
                    agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)
                    
                    
        
        '''save'''
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
        
    cur_logger.log("test end!")

