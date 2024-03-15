#!/bin/bash
#  ppo_mh  N vs 1 mbam_om_mh
# Begin experiment#
python3.6 main.py \
--exp_name "simple_tag_ppo_vs_mbam_10oppo" \
--env "simple_tag" \
--prefix "train" \
--train_mode 0 \
--seed -1 \
--ranks 1 \
--device "cuda:0" \
--dir "" \
--eps_max_step 50 \
--eps_per_epoch 10 \
--save_per_epoch 100 \
--max_epoch 100000 \
--true_prob True \
#--actor_rnn True \
#--prophetic_onehot False \


#--continue_train True \  #需要修改load文件