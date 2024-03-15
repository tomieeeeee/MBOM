import json
import os
import time
import atexit
import numpy as np
import re
from torch.utils.tensorboard import SummaryWriter
from DRL_MARL_homework.MBAM.utils.plot_util import symmetric_ema
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, dir, exp_name, seed):
        '''
        :param log_dir: save root dir or saved log dir
        :param exp_name: is None when load mode. different random seed use same exp_name
        :param load_exist_log:
        '''
        '''
        root_dir/exp_name/seed/log/ SummaryWriter file
        root_dir/exp_name/seed/model/  "model.name_iter{}.ckp".format(iteration)
        root_dir/exp_name/seed/param.json(2-conf, args, seed)
        root_dir/exp_name/seed/log.txt
        '''
        self.dir = dir
        self.exp_name = exp_name
        self.seed = seed
        try:
            idx = max(list(map(int, [d.split("_")[0] for d in os.listdir(os.path.join(dir, exp_name))]))) + 1
        except:
            idx = 0
        self.root_dir = os.path.join(dir, exp_name, "{}_{}".format(idx, seed))
        assert not os.path.exists(self.root_dir), "log dir is exist! {}".format(self.root_dir)
        os.makedirs(self.root_dir)
        self.model_dir = os.path.join(self.root_dir, "model")
        os.makedirs(self.model_dir)
        self.log_dir = os.path.join(self.root_dir, "log")
        self.log_file = open(os.path.join(self.root_dir, "log.txt"), 'a')
        atexit.register(self.log_file.close)
        self.param_file =os.path.join(self.root_dir, "param.json")
        self.writer = SummaryWriter(self.log_dir)
        atexit.register(self.writer.flush)
        atexit.register(self.writer.close)
        pass

    def log_performance(self, tag, iteration, **kwargs):
        for k, v in kwargs.items():
            main_tag = tag + "/" + k
            self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict={k: v}, global_step=iteration)
        self.writer.flush()

    def log_param(self, args, confs, **kwargs):
        with open(self.param_file, 'a') as f:
            f.write(json.dumps(kwargs))
            f.write(json.dumps({"args": vars(args), "confs": confs}))
        self.writer.flush()

    def log(self, s):
        print(s)
        self.log_file.write(s + "\n")
        self.log_file.flush()

    @staticmethod
    def load_log_to_numpy(exp_path, keys):
        '''
        :param exp_path: exp dir list, contains different opponent
        :param keys: list
        :return: dict("steps":..., "epoch":...)
                 Agent1_Score: np.ndarray [n_seed, n_opponent, n_episode]
        '''
        from tensorboard.backend.event_processing import event_accumulator
        def get_logfile_path(log_dir):
            events_dir_list = []
            for root, dirs, files in os.walk(log_dir):  # + "/0_1477287147/log"
                for f in files:
                    if f.startswith("events.out") and (not root.endswith("log")):
                        events_dir_list.append(root)
            return events_dir_list
        def get_seed_data(log_dir, seed_dir, keys):
            '''
            :param log_dir: exp dir
            :param seed_dir: list, seed_dir
            :param keys: list
            :return:dict("steps":..., "epochs":...)
                    Agent1_Score: np.ndarry [n_opponent, n_episode]
            '''
            # epochs_key = ["Entropy", "KL", "Loss_a", "Loss_v", "Score"]
            # steps_key = ["Loss_oppo", "Mix_entropy", "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"]
            epochs_key = ["Entropy", "KL", "Loss_a", "Loss_v", "Score"]
            steps_key = ["Loss_oppo", "Mix_entropy", "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"]
            events_dir_list = get_logfile_path(os.path.join(log_dir, seed_dir))
            log_data = {}
            # get all log_data
            for path in events_dir_list:
                # if not need log then ignore it
                if keys == None:
                    pass
                else:
                    flag = False
                    for k in keys:
                        if k in path:
                            flag = True
                            break
                    if not flag:
                        continue

                ea = event_accumulator.EventAccumulator(path)
                ea.Reload()
                for log_full_key in ea.scalars.Keys():
                    val_psnr = ea.scalars.Items(log_full_key)
                    if log_full_key not in log_data.keys():
                        log_data[log_full_key] = []
                    temp = [x[1] for x in sorted([(i.step, i.value) for i in val_psnr], key=lambda x: x[0])]
                    log_data[log_full_key].append(temp)

            # data to prepare
            output_data = {"steps": {},
                           "epochs": {}}
            for log_key in log_data.keys():
                if "rank" in log_key:
                    continue
                for data_key in epochs_key:
                   if data_key in log_key:
                       min_step = min([len(log_data[log_key][i]) for i in range(len(log_data[log_key]))])
                       output_data["epochs"][data_key] = np.array([a[:min_step] for a in log_data[log_key]])
                for data_key in steps_key:
                    if data_key in log_key:
                        min_step = min([len(log_data[log_key][i]) for i in range(len(log_data[log_key]))])
                        output_data["steps"][data_key] = np.array([a[:min_step] for a in log_data[log_key]])
            return output_data
        def get_exp_data(log_dir, keys):
            '''
            :param log_dir: exp dir
            :param keys: list
            :return: dict("steps":..., "epoch":...)
                     Agent1_Score: np.ndarry [n_seed, n_opponent, n_episode]
            '''
            seed_dir = os.listdir(log_dir)
            output_data = {"epochs": {},
                           "steps": {}}
            seed_datum = []
            for s in seed_dir:
                seed_data = get_seed_data(log_dir, s, keys)
                seed_datum.append(seed_data)
            for xtick in ["epochs", "steps"]:
                for k in seed_datum[0][xtick].keys():
                    temp = [data[xtick][k] for data in seed_datum]
                    min_step = min([t.shape[1] for t in temp])
                    temp = [t[:, :min_step] for t in temp]
                    if k not in output_data[xtick].keys():
                        output_data[xtick][k] = []
                    output_data[xtick][k] = np.stack(temp)
            return output_data
        def merged_exp_data(exp_paths, keys, merge_type=""):
            '''
            :param exp_paths:
            :param keys:
            :param merge_type: "cat" or "stack"
            :return:
            '''
            assert type(exp_path) is list, "type(exp_path) must be list"

            datum = []
            output_data = {"epochs": {},
                           "steps": {}}
            for p in exp_paths:
                datum.append(get_exp_data(p, keys))
            for xtick in ["epochs", "steps"]:
                for keys in datum[0][xtick].keys():
                    temp = [data[xtick][keys] for data in datum]
                    min_step = min([t.shape[2] for t in temp])
                    temp = [t[:, :, :min_step] for t in temp]
                    if merge_type == "cat":
                        output_data[xtick][keys] = np.concatenate(temp, axis=1)
                    elif merge_type == "stack":
                        output_data[xtick][keys] = np.stack(temp, axis=1)
            return output_data
        all_data = merged_exp_data(exp_path, keys, merge_type="cat")
        return all_data

    @staticmethod
    def plot_log(exp_paths, keys):
        '''
        :param exp_paths: list [ [], [] ]
                     Agent1_Score: np.ndarray [n_seed, n_opponent, n_episode]
        :param keys:
        :return:
        '''


        epochs_key = ["Entropy", "KL", "Loss_a", "Loss_v", "Score"]
        steps_key = ["Loss_oppo", "Mix_entropy", "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"]
        colors = ['#0072B2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf', '#8c564b']

        def plot_curves(x, y, color=None, title=None, xlabel=None, ylabel=None, is_plot_std=False, std_type="std"):
            pass


        pass
    @staticmethod
    def plot_curve(x, y, color=None, title=None, xlabel=None, ylabel=None, is_plot_std=False, std_type="std"):
        '''
        :param x: np.ndarray [n_seeds, n_xticks]
        :param y: np.ndarray [n_seeds, n_xticks]
        :param color:
        :param title:
        :param xlabel:
        :param ylabel:
        :param std_type: "std" or "minmax"
        :return:
        '''
        xs = []
        ys = []
        for t in y:
            x_, y_, _ = symmetric_ema(x, t, low=x[0], high=x[-1], n=300)
            xs.append(x_)
            ys.append(y_)
        xs = np.mean(xs, axis=0)
        ymean = np.mean(ys, axis=0)
        ymin = np.min(ys, axis=0)
        ymax = np.max(ys, axis=0)
        ystd = np.std(ys, axis=0)
        plt.plot(xs, ymean, color=color)
        if is_plot_std:
            if std_type == "std":
                plt.fill_between(xs, ymean - ystd, ymean + ystd, color=color, alpha=.2)
            elif std_type == "minmax":
                plt.fill_between(xs, ymin, ymax, color=color, alpha=.2)
            else:
                print("std_type must be 'std' or 'minmax'")
                raise NameError
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.show()

    @staticmethod
    def plot_Score(datum, is_sort=False):
        '''
        :param data: np.ndarray [n_seed, n_opponent, n_episode]
        :return:
        '''
        colors = ['#0072B2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf', '#8c564b']
        for i, data in enumerate(datum):
            y = data["epochs"]["Score"].mean(axis=2)
            x = np.arange(y.shape[1])
        pass

if __name__ == "__main__":
    # logger = Logger("./", "1v1", 123)
    # record = {"train/123":123,
    #           "train/234":123}
    # for i in range(100):
    #     logger.log_performance("PPO/loss/a_loss", i, **record)
    #score [n_seed, n_opponent, n_episode]
    '''
    score [n_seed, n_opponent, n_episode]
    '''
    colors = ['#0072B2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#8c564b']
    import matplotlib.pyplot as plt
    import numpy as np

    for i, c in enumerate(colors):
        plt.plot(np.arange(100), [i+1] * 100, color=colors[i])
    plt.show()

    #from tensorboard.backend.event_processing import event_accumulator
    #path = "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/rnn_football_ppo_vs_mbam/7_1334749859/log/MBAM_football_epochs_Entropy_Entropy/"
    log_dir = ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_0_vs_mbam",
               "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_1_vs_mbam",
               "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_mbam_0_vs_mbam",]
    #keys = ["Loss_oppo", "Mix_entropy", "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio", "Entropy", "KL", "Loss_a", "Loss_v", "Score"]

    all_data = Logger.load_log_to_numpy(log_dir, ["Score"])
    a = 1
    #all_data = get_exp_data(log_dir, keys)


    # events_dir_list = []
    # for root, dirs, files in os.walk(log_dir): # + "/0_1477287147/log"
    #     for f in files:
    #         if f.startswith("events.out") and (not root.endswith("log")):
    #             events_dir_list.append(root)
    # data = {}
    # for path in events_dir_list:
    #     ea = event_accumulator.EventAccumulator(path)
    #     ea.Reload()
    #     #print(ea.scalars.Keys())
    #     for key in ea.scalars.Keys():
    #         val_psnr = ea.scalars.Items(key)
    #         if key not in data.keys():
    #             data[key] = []
    #         temp = [x[1] for x in sorted([(i.step, i.value) for i in val_psnr], key=lambda x:x[0])]
    #         data[key].append(temp)
    #
    #     # val_psnr = ea.scalars.Items('val_psnr')
    #     # print(len(val_psnr))
    #     # print([(i.step, i.value) for i in val_psnr])
    # data = data
    # for key in data.keys():
    #     if key.startwith("MBAM") and key.endwith("Score"):
    #         score_list = data[key]

    '''
    def save_tag_to_csv(fn, tag='test_metric', output_fn=None):
        if output_fn is None:
            output_fn = '{}.csv'.format(tag.replace('/', '_'))
        print("Will save to {}".format(output_fn))
    
        sess = tf.InteractiveSession()
    
        wall_step_values = []
        with sess.as_default():
            for e in tf.train.summary_iterator(fn):
                for v in e.summary.value:
                    if v.tag == tag:
                        wall_step_values.append((e.wall_time, e.step, v.simple_value))
        np.savetxt(output_fn, wall_step_values, delimiter=',', fmt='%10.5f')
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('fn')
        parser.add_argument('--tag', default='test_metric')
        args = parser.parse_args()
        save_tag_to_csv(args.fn, tag=args.tag)
    '''