# RL model and buffer
from policy.MBAM import MBAM
from policy.MBAM_OM_MH import MBAM_OM_MH
from policy.MBAM_MH import MBAM_MH
from baselines.PPO import PPO, PPO_Buffer
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
from baselines.PPO_OM_MH import PPO_OM_MH, PPO_OM_MH_Buffer

# env
from env_wapper.simple_tag.simple_tag import Simple_Tag
# env_model
from env_model.simple_tag.model_simple_tag import load_env_model as simple_tag_env_model
# conf
from config.simple_tag_conf import player1_conf, player2_conf
from utils.rl_utils_MH import collect_trajectory_MH, collect_trajectory_MH_reversed
from utils.rl_utils import collect_trajectory
# logger
from utils.Logger import Logger
from utils.get_process_memory import get_processes_memory_gb
import random
import torch
import numpy as np
import multiprocessing as mp
import os
from utils.get_exp_data_path import get_exp_data_path

def simple_tag_trainer(args, logger):
    mp.set_start_method("spawn")
    if args.train_mode==0:
        channel_in = [mp.Queue(maxsize=1) for _ in range(args.ranks)]
        channel_out = [mp.Queue(maxsize=1) for _ in range(args.ranks)]
        if args.continue_train:
            """continue train"""
            continue_train_path = get_exp_data_path() + "/Simple_RPS/Player2/continue_mbam/MBAM_trigame2_iter35600.ckp"
            global_mbam = MBAM.load_model(continue_train_path, args, logger=logger, device=args.device, env_model=None)
        else:
            """normal train"""
            env_model = simple_tag_env_model(args.device)
            global_mbam = MBAM_OM_MH(args=args, conf=player2_conf, name="simpletag2", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
        global_buffer = PPO_OM_MH_Buffer(args=args, conf=global_mbam.conf, name=global_mbam.name, actor_rnn=args.actor_rnn, device=args.device)
        processes = []
        for rank in range(args.ranks):
            #新建一个线程
            p = mp.Process(target=worker, args=(args, logger.root_dir, rank, channel_in[rank], channel_out[rank]))
            #开始执行worker函数
            p.start()
            #线程p加入processes列表
            processes.append(p)
        pid_list = [p.pid for p in processes] + [os.getpid()]
        #开始训练
        for epoch in range(1, args.max_epoch + 1):
            for rank in range(args.ranks):
                #主进程向子进程发送a-net信息,由于子进程10个epoch才接收数据
                channel_out[rank].put({"a_net": global_mbam.a_net.state_dict(), "v_net": global_mbam.v_net.state_dict()})
            logger.log("globallllll, epoch:{} param shared!".format(epoch))
            datum = []
            for rank in range(args.ranks):
                #接收子进程的数据
                data = channel_in[rank].get()
                datum.append(data)
            merge_data = dict()
            for key in data.keys():
                if type(data[key]) is torch.Tensor:
                    merge_data[key] = torch.cat([d[key] for d in datum])
                elif type(data[key]) is list:
                    merge_data[key] = []
                    for i in range(len(data[key])):
                        merge_data[key].append(torch.cat([d[key][i] for d in datum]))
                    #for d in datum:
                    #    merge_data[key] = merge_data[key] + d[key]
                else:
                    raise TypeError
            #绿学
            global_mbam.learn(data=merge_data, iteration=epoch, no_log=False)
            logger.log("global, epoch:{} param updated!".format(epoch, epoch))
            if epoch % args.save_per_epoch == 0:
                    global_mbam.save_model(epoch)
            # logger.log("memory:{}".format(get_processes_memory_gb(pid_list)))
        for p in processes:
            p.join()
        pass
    elif args.train_mode == 1:
        individual_worker(args, logger)
    elif args.train_mode == 2:
        individual_worker_reversed(args, logger)
    elif args.train_mode == 3:
        continue_train_path = get_exp_data_path() + "/Simple_RPS/new_Player2/MBAM_trigame2_iter100000.ckp"
        global_mbam = MBAM.load_model(continue_train_path, args, logger=logger, device=args.device, env_model=None)
        continue_train_worker(args, logger, global_mbam)
    else:
        print("train_model error")
        raise NameError

def worker(args, root_dir, rank, channel_out, channel_in):
    env = Simple_Tag(args.eps_max_step)
    env_model = simple_tag_env_model(args.device)
    logger = Logger(root_dir, "worker", rank)
    ppo = PPO_MH(args, player1_conf, name="player1_rank{}".format(rank), logger=logger, actor_rnn=args.actor_rnn, device=args.device)
    mbam = MBAM_OM_MH(args=args, conf=player2_conf, name="player2", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
    agents = [ppo, mbam]
    buffers = [PPO_MH_Buffer(args=args, conf=agents[0].conf, name=agents[0].name, actor_rnn=args.actor_rnn, device=args.device),
               PPO_OM_MH_Buffer(args=args, conf=agents[1].conf, name=agents[1].name, actor_rnn=args.actor_rnn, device=args.device)]
    logger.log_param(args, [agent.conf for agent in agents], rank=rank)
    global_step = 0
    for epoch in range(1, args.max_epoch + 1):
        """隔几次update一次param"""
        if epoch % 10 == 1:
            #接收主进程的数据
            param = channel_in.get()
            mbam.a_net.load_state_dict(param["a_net"])
            mbam.v_net.load_state_dict(param["v_net"])

        logger.log("rankkkkkk:{}, epoch:{} start!".format(rank, epoch))
        #rl_utils_MH.py的方法collect_trajectory_MH，会收集eps_per_epoch个回合的数据
        memory, scores, global_step, touch_times = collect_trajectory_MH(agents, env, args, global_step, is_prophetic=True)
        for i in range(2):
            logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i], Touch=touch_times)
            assert args.env == "simple_tag", "env is not simple_tag"
            last_val = [agents[i].v_net(torch.from_numpy(m.final_state[0].astype(np.float32)).to(args.device)).detach().cpu().numpy().item() for m in memory[i]]
            buffers[i].store_multi_memory(memory[i], last_val=last_val)
            #buffers[i].store_multi_memory(memory[i], last_val=0)
        #只有红方在改进
        agents[0].learn(data=buffers[0].get_batch(), iteration=epoch, no_log=False)

        """隔几次update一次param"""
        if epoch % 10 == 0:
            #10个epoch之后向主进程发送数据
            channel_out.put(buffers[1].get_batch())
        #红绿双方保存模型
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
    print("end")

def individual_worker(args, logger, **kwargs):
    '''set seed'''
    random.seed(logger.seed)
    torch.manual_seed(logger.seed)
    np.random.seed(logger.seed)
    '''env'''
    env = Simple_Tag(eps_max_step=args.eps_max_step)
    env_model = None
    '''prepare agents'''
    
    ppo = PPO(args, player1_conf, name="player1", logger=logger, actor_rnn=args.actor_rnn, device=args.device)
    
    mbam = MBAM(args=args, conf=player2_conf, name="player2", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn,
                env_model=env_model, device=args.device)

    agents = [ppo, mbam]
    buffers = [PPO_Buffer(args=args, conf=agent.conf, name=agent.name, actor_rnn=args.actor_rnn, device=args.device) for agent in agents]

    logger.log_param(args, [agent.conf for agent in agents])
    global_step = 0
    for epoch in range(1, args.max_epoch + 1):
        logger.log("epoch:{} start!".format(epoch))
        memory, scores, global_step = collect_trajectory(agents, env, args, global_step, is_prophetic=True)
        for i in range(2):
            logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i])
            if args.alter_train:
                if int((epoch - 1) / args.alter_interval) % 2 == i:
                    buffers[i].store_multi_memory(memory[i], last_val=0)
                    agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)
                else:
                    buffers[i].clear_memory()
            else:
                buffers[i].store_multi_memory(memory[i], last_val=0)
                agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
        # logger.log("memory:{}".format(get_current_memory_gb()))
    logger.log("train end!")


def individual_worker_reversed(args, logger, **kwargs):
    '''set seed'''
    random.seed(logger.seed)
    torch.manual_seed(logger.seed)
    np.random.seed(logger.seed)
    '''env'''
    env = Simple_Tag(eps_max_step=args.eps_max_step)
    env_model = None
    '''prepare agents'''
    player1 = MBAM_MH(args=args, conf=player1_conf, name="player1", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
    player2 = PPO(args, player2_conf, name="player2", logger=logger, actor_rnn=args.actor_rnn, device=args.device)
    agents = [player1, player2]

    player1_buffer = PPO_MH_Buffer(args, player1_conf, player1.name, actor_rnn=args.actor_rnn, device=args.device)
    player2_buffer = PPO_OM_MH_Buffer(args, player2_conf, player2.name, actor_rnn=args.actor_rnn, device=args.device)
    buffers = [player1_buffer, player2_buffer]

    logger.log_param(args, [agent.conf for agent in agents])
    global_step = 0
    for epoch in range(1, args.max_epoch + 1):
        logger.log("epoch:{} start!".format(epoch))
        memory, scores, global_step, touch_times = collect_trajectory_MH_reversed(agents, env, args, global_step, is_prophetic=True)
        for i in range(2):
            logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i], Touch=touch_times)
            if args.alter_train:
                raise NotImplementedError
                if int((epoch - 1) / args.alter_interval) % 2 == i:
                    buffers[i].store_multi_memory(memory[i], last_val=0)
                    agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)
                else:
                    buffers[i].clear_memory()
            else:
                assert args.env == "simple_tag", "env is not simple_tag"
                last_val = [agents[i].v_net(torch.from_numpy(m.final_state[0].astype(np.float32)).to(args.device)).detach().cpu().numpy().item() for m in memory[i]]
                buffers[i].store_multi_memory(memory[i], last_val=last_val)
                #buffers[i].store_multi_memory(memory[i], last_val=0)
                agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
        # logger.log("memory:{}".format(get_current_memory_gb()))
    logger.log("train end!")


if __name__ == "__main__":
    from main import main
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--exp_name", type=str, default="simple_tag_mbam_ppo", help="football_ppo_vs_mbam\n " +
                                                                                          "predator_ppo_vs_mbam\n" +
                                                                                          "rps_ppo_vs_mbam\n" +
                                                                                          "simple_push_ppo_vs_mbam\n" +
                                                                                          "trigame_ppo_vs_mbam\n" +
                                                                                          "simple_rps_ppo_vs_mbam\n" +
                                                                                          "coin_game_ppo_vs_mbam\n" +
                                                                                          "simple_tag_ppo_mbam\n")
    parser.add_argument("--env", type=str, default="simple_tag", help="football\n" +
                                                                   "predator\n" +
                                                                   "rps\n" +
                                                                   "simple_push\n" +
                                                                   "trigame\n" +
                                                                   "simple_rps\n" +
                                                                   "coin_game\n" +
                                                                   "simple_tag\n")

    parser.add_argument("--prefix", type=str, default="train", help="train or test or search")

    parser.add_argument("--train_mode", type=int, default=2,
                        help="0 1 2 3means N vs 1 de ppo vs mbam, mbamvsppo, continue_train")
    parser.add_argument("--alter_train", type=int, default=0, help="0 1 means no and yes")
    parser.add_argument("--alter_interval", type=int, default=100, help="epoch")
    parser.add_argument("--continue_train", type=bool, default=False, help="0 1 means no and yes")
    parser.add_argument("--batch_size", type=int, default=2, help="")

    parser.add_argument("--test_mode", type=int, default=0, help="0 1 2 means layer0, layer1, layer2")
    parser.add_argument("--test_mp", type=int, default=1, help="multi processing")

    parser.add_argument("--seed", type=int, default=-1, help="-1 means random seed")
    parser.add_argument("--ranks", type=int, default=2, help="for prefix is train")
    parser.add_argument("--device", type=str, default="cpu", help="")
    parser.add_argument("--dir", type=str, default="", help="")

    parser.add_argument("--eps_max_step", type=int, default=30, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=10, help="")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="")
    parser.add_argument("--max_epoch", type=int, default=100, help="train epoch")
    parser.add_argument("--num_om_layers", type=int, default=3, help="train epoch")
    parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")

    parser.add_argument("--actor_rnn", type=bool, default=False, help="True or False")
    parser.add_argument("--true_prob", type=bool, default=True, help="True or False, edit Actor_RNN.py line 47-48")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--policy_training", type=bool, default=False, help="True or False")

    parser.add_argument("--only_use_last_layer_IOP", type=bool, default=False, help="True or False")
    parser.add_argument("--random_best_response", type=bool, default=False, help="True or False")

    parser.add_argument("--record_more", type=bool, default=False, help="True or False")
    parser.add_argument("--config", type=str, default="", help="extra info")
    args = parser.parse_args()
    main(args)