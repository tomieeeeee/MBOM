import argparse
import random
import time
import os
import sys
sys.path.append("/home/lenovo/文档/CodeWorkspace/RL")
sys.path.append("/home/xiaopeng/CodeWorkspace/RL")
# logger
from utils.Logger import Logger
# trainer tester
from exp.simple_tag.trainer import simple_tag_trainer
from exp.simple_tag.tester import simple_tag_tester
#from exp.coin_game.trainer import coin_game_trainer
#from exp.coin_game.tester import coin_game_tester
# dir
from utils.get_exp_data_path import get_exp_data_path
# env_model
from env_model.simple_tag.model_simple_tag import ENV_Simple_Tag
import os
import psutil
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


def main(args):
    seed = random.randint(0, int(time.time())) if args.seed == -1 else args.seed
    dir = get_exp_data_path() if args.dir == "" else args.dir

    if args.env == "simple_tag":
        dir = os.path.join(dir, "Simple_Tag")
        #重命名simple_tag_trainer/tester方法
        trainer = simple_tag_trainer
        tester = simple_tag_tester
    else:
        raise NameError
    exp_name = "{}/{}{}{}{}".format(args.prefix,
                                    ("pone_" if args.prophetic_onehot else ""),
                                    ("trueprob_" if args.true_prob else ""),
                                    ("rnn_" if args.actor_rnn else ""),
                                    args.exp_name)
    logger = Logger(dir, exp_name, seed)
    if args.prefix == "train":
        trainer(args, logger)
    elif args.prefix == "test":
        tester(args, logger)

if __name__ == "__main__":
    #graphviz = GraphvizOutput()
    #graphviz.output_file = 'bbasic.png'
   # with PyCallGraph(output=graphviz):
      parser = argparse.ArgumentParser(description="")

      parser.add_argument("--exp_name", type=str, default="trigame_ppo_0_vs_mbam_M_2", help="football_ppo_vs_mbam\n " +
                                                                                     "predator_ppo_vs_mbam\n" +
                                                                                     "rps_ppo_vs_mbam\n" +
                                                                                     "simple_push_ppo_vs_mbam\n" +
                                                                                     "trigame_ppo_vs_mbam\n" +
                                                                                     "simple_rps_ppo_vs_mbam\n" +
                                                                                     "coin_game_ppo_vs_mbam\n" +
                                                                                     "simple_tag_ppo_mbam\n")
      parser.add_argument("--env", type=str, default="trigame", help="football\n" +
                                                                   "predator\n" +
                                                                   "rps\n" +
                                                                   "simple_push\n" +
                                                                   "trigame\n" +
                                                                   "simple_rps\n" +
                                                                   "coin_game\n" +
                                                                   "simple_tag\n")

      parser.add_argument("--prefix", type=str, default="test", help="train or test or search")

      parser.add_argument("--train_mode", type=int, default=1, help="0 1 2 3means N vs 1 de ppo vs mbam, mbamvsppo, continue_train")
      parser.add_argument("--alter_train", type=int, default=0, help="0 1 means no and yes")
      parser.add_argument("--alter_interval", type=int, default=100, help="epoch")
      parser.add_argument("--continue_train", type=bool, default=False, help="0 1 means no and yes")
      parser.add_argument("--batch_size", type=int, default=2, help="")

      parser.add_argument("--test_mode", type=int, default=0, help="0 1 2 means layer0, layer1, layer2")
      parser.add_argument("--test_mp", type=int, default=1, help="multi processing")
      parser.add_argument("--player2_is_ppo", type=bool, default=False, help="")

      parser.add_argument("--seed", type=int, default=-1, help="-1 means random seed")
      parser.add_argument("--ranks", type=int, default=1, help="for prefix is train")
      parser.add_argument("--device", type=str, default="cuda", help="")
      parser.add_argument("--dir", type=str, default="", help="")

      parser.add_argument("--eps_max_step", type=int, default=30, help="")
      parser.add_argument("--eps_per_epoch", type=int, default=10, help="")
      parser.add_argument("--save_per_epoch", type=int, default=100, help="")
      parser.add_argument("--max_epoch", type=int, default=100, help="train epoch")
      parser.add_argument("--num_om_layers", type=int, default=3, help="train epoch")
      parser.add_argument("--agent1_num_om_layers", type=int, default=1, help="train epoch")
      parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")

      parser.add_argument("--actor_rnn", type=bool, default=False, help="True or False")
      parser.add_argument("--true_prob", type=bool, default=False, help="True or False, edit Actor_RNN.py line 47-48")
      parser.add_argument("--prophetic_onehot", type=bool, default=False, help="for training, True or False")
      parser.add_argument("--policy_training", type=bool, default=False, help="True or False")
      #policy相关参数
      parser.add_argument("--only_use_last_layer_IOP", type=bool, default=False, help="True or False")
      parser.add_argument("--random_best_response", type=bool, default=False, help="True or False")
      parser.add_argument("--uniform_mixing", type=bool, default=False, help="True or False")
      parser.add_argument("--softupdate_alpha", type=bool, default=False, help="True or False")
      parser.add_argument("--softupdate_alpha_lr", type=float, default=0.1, help="True or False")
      parser.add_argument("--prophetic_action_probs", type=bool, default=False, help="for Test, True or False")
      parser.add_argument("--cal_action_probs_KL_divengence", type=bool, default=False, help="for Test, prophetic_action_probs must be true ,True or False")

      parser.add_argument("--record_more", type=bool, default=False, help="True or False")
      parser.add_argument("--config", type=str, default="", help="extra info")
      args = parser.parse_args()
      main(args)