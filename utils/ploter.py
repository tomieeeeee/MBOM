import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.plot_util import symmetric_ema
import matplotlib.pyplot as plt
import math

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
            # if ("shooter" in path) or ("runner" in path):   #将进攻方屏蔽掉
            #     continue
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
            # if ("shooter" in log_key) or ("runner" in log_key) or ("rank" in log_key):   #将进攻方屏蔽掉
            #     continue
            # if ("MBAM" in log_key):
            #     continue
            if ("goalkeeper" not in log_key) and ("predator" not in log_key) and ("meta-agent" not in log_key) and ("player2" not in log_key):   #将进攻方屏蔽掉
                continue
            for data_key in epochs_key:
                if data_key in log_key:
                    # if data_key == "Score" and len(log_data[log_key][0]) > 100:
                    #     for j in range(len(log_data[log_key])):
                    #         log_data[log_key][j] = [d for i, d in enumerate(log_data[log_key][j]) if i % 2 == 1]
                    min_step = min([len(log_data[log_key][i]) for i in range(len(log_data[log_key]))])
                    output_data["epochs"][data_key] = np.array([a[:min_step] for a in log_data[log_key]])
                    if "meta" in log_key:
                        output_data["epochs"][data_key] = -1 * output_data["epochs"][data_key]
            for data_key in steps_key:
                if data_key in log_key:
                    min_step = min([len(log_data[log_key][i]) for i in range(len(log_data[log_key]))])
                    output_data["steps"][data_key] = np.array([a[:min_step] for a in log_data[log_key]])
                    if "meta" in log_key:
                        output_data["steps"][data_key] = -1 * output_data["steps"][data_key]
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

def plot_curve(ax, x, y, color=None, title=None, xlabel=None, ylabel=None, is_plot_std=False, std_type="std"):
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
    if ax is None:
        drawer = plt
    else:
        drawer = ax
    xs = []
    ys = []
    for t in y:
        x_, y_, _ = symmetric_ema(x, t, low=x[0], high=x[-1], n=90)
        xs.append(x_)
        ys.append(y_)
    xs = np.mean(xs, axis=0)
    ymean = np.mean(ys, axis=0)
    ymin = np.min(ys, axis=0)
    ymax = np.max(ys, axis=0)
    ystd = np.std(ys, axis=0)
    drawer.plot(xs, ymean, color=color)
    if is_plot_std:
        if std_type == "std":
            drawer.fill_between(xs, ymean - ystd, ymean + ystd, color=color, alpha=.2)
        elif std_type == "minmax":
            drawer.fill_between(xs, ymin, ymax, color=color, alpha=.2)
        else:
            print("std_type must be 'std' or 'minmax'")
            raise NameError
    if ax is None:
        drawer.title(title)
        drawer.xlabel(xlabel)
        drawer.ylabel(ylabel)
    else:
        drawer.set_title(title)
        drawer.set_xlabel(xlabel)
        drawer.set_ylabel(ylabel)
    # plt.show()

def plot_Score(datum, is_sort=False, interval=None, curve_name=None):
    '''
    :param data: np.ndarray [n_seed, n_opponent, n_episode]
    :return:
    '''
    colors = ['#0072B2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf', '#8c564b']
    temp_result = []
    if interval == None:
        for i, data in enumerate(datum):
            y = data["epochs"]["Score"].mean(axis=2)
            x = np.arange(y.shape[1])
            if is_sort:
                y.sort()
            plot_curve(None, x, y, colors[i % len(colors)], title="Score", xlabel="epoch", ylabel="ave_reward", is_plot_std=True, std_type="std")
            plt.legend(curve_name)
            temp_result.append(y.mean())
    else:
        fig = plt.figure(figsize=(20, 15))
        N_fig_each_row = 3
        n_sub_fig = int((datum[0]["epochs"]["Score"].shape[-1] + interval - 1) / interval)
        axes = []
        for j in range(n_sub_fig):
            axes.append(fig.add_subplot(math.ceil(n_sub_fig / N_fig_each_row), N_fig_each_row, j + 1))
        for i, data in enumerate(datum):
            all_y = data["epochs"]["Score"]
            for j in range(n_sub_fig):
                if j != n_sub_fig - 1:
                    y = all_y[:, :, j * interval:(j + 1) * interval].mean(axis=2)
                else:
                    y = all_y[:, :, j * interval:].mean(axis=2)
                x = np.arange(y.shape[1])
                if is_sort:
                    y.sort()
                plot_curve(axes[j], x, y, colors[i % len(colors)], title="Score-{}".format(j), xlabel="epoch", ylabel="ave_reward", is_plot_std=True, std_type="std")
        fig.legend(curve_name, loc='lower right')
    plt.show()

if __name__ == "__main__":
    # log_dir = [
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_ppo_0_vs_lola_2",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_ppo_1_vs_lola_2",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_mbam_0_vs_lola_2"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_ppo_0_vs_meta_oppo",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_ppo_1_vs_meta_oppo",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_mbam_0_vs_meta_oppo"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_ppo_0_vs_meta",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_ppo_1_vs_meta",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/football_mbam_0_vs_meta"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_0_vs_mbam",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_1_vs_mbam",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_mbam_0_vs_mbam"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_0_vs_mbam_1",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_1_vs_mbam_1",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_mbam_0_vs_mbam_1"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_0_vs_mbam_1_notrain",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_1_vs_mbam_1_notrain",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_mbam_0_vs_mbam_1_notrain"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_0_vs_mbam_2",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_1_vs_mbam_2",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_mbam_0_vs_mbam_2"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_0_vs_mbam_notrain",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_ppo_1_vs_mbam_notrain",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Football_Penalty_Kick/test/trueprob_rnn_football_mbam_0_vs_mbam_notrain"],
    # ]
    # curve_name = ["lola", "meta_oppo", "meta", "my_method", "1_layers_om", "1_layers_om_notrain", "2_layers_om", "mbam_notrain"]
    # all_data = []
    # for dir in log_dir:
    #     all_data.append(load_log_to_numpy(dir, ["Score"]))#, "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"
    # all_data[4]["epochs"]["Score"][:, 60:, :] -= 0
    # plot_Score(all_data, is_sort=False, interval=10, curve_name=curve_name)
    # plot_Score(all_data, is_sort=True, interval=None, curve_name=curve_name)
    # plot_Score(all_data, is_sort=False, interval=None, curve_name=curve_name)

    #查看simple-predator结果
    # log_dir = [
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test1/trueprob_predator_ppo_0_vs_mbam",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test1/trueprob_predator_ppo_1_vs_mbam",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test1/trueprob_predator_mbam_0_vs_mbam"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test/predator_ppo_0_vs_meta_oppo",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test/predator_ppo_1_vs_meta_oppo",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test/predator_mbam_0_vs_meta_oppo"],
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test/predator_ppo_0_vs_meta",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test/predator_ppo_1_vs_meta",
    #      "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/test/predator_mbam_0_vs_meta"],
    # ]
    # curve_name = ["my_method", "meta_oppo", "meta"]
    # all_data = []
    # for dir in log_dir:
    #     all_data.append(load_log_to_numpy(dir, ["Score"]))  #, "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"
    # plot_Score(all_data, is_sort=False, interval=10, curve_name=curve_name)
    # plot_Score(all_data, is_sort=True, interval=None, curve_name=curve_name)
    # plot_Score(all_data, is_sort=False, interval=None, curve_name=curve_name)

    # 查看simple_rps结果
    log_dir = [
        # ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_ppo_0_vs_meta_oppo",
        #  "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_ppo_1_vs_meta_oppo",
        #  "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_mbam_0_vs_meta_oppo"],
        # ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_ppo_0_vs_meta",
        #  "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_ppo_1_vs_meta",
        #  "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_mbam_0_vs_meta"],
        # ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_ppo_0_vs_lola",
        #  "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_ppo_1_vs_lola",
        #  "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/simple_rps_mbam_0_vs_lola"],
        ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/trueprob_simple_rps_ppo_0_vs_mbam_om2",
         "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/trueprob_simple_rps_ppo_1_vs_mbam_om2",
         "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/trueprob_simple_rps_mbam_1_vs_mbam_om2"],
        ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/trueprob_simple_rps_ppo_0_vs_mbam_om1",
         "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/trueprob_simple_rps_ppo_1_vs_mbam_om1",
         "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_RPS/test/trueprob_simple_rps_mbam_1_vs_mbam_om1", ],
    ]
    curve_name = ["meta_oppo", "meta", "lola", "mbam_om2", "mbam_om1"]
    all_data = []
    for dir in log_dir:
        all_data.append(
            load_log_to_numpy(dir, ["Score"]))  # , "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"

    # 查看search结果
    #path = "/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/search/trueprob_predator_ppo_1_vs_mbam/"
    #path_list = []
    #path_list = os.listdir(path)
    #path_list = [path + p for p in path_list]
    # log_dir = [
    #     ["/media/lenovo/144ED9814ED95C54/experiment_data/Simple_Predator/search/trueprob_predator_ppo_1_vs_mbam/%s" % str(i)] for i in range(60)#[2, 10, 51]
    #     #path_list
    # ]
    # # [2, 3, 4, ]
    # all_data = []
    # for dir in log_dir:
    #     all_data.append(load_log_to_numpy(dir, ["Score", "om_layer0_mix_ratio", "om_layer1_mix_ratio", "om_layer2_mix_ratio"]))
    # plot_Score(all_data, is_sort=False, interval=None)
