#!/bin/bash
#  ppo_mh  N vs 1 mbam_om_mh
# Begin experiment#
python main.py \
--exp_name "simple_tag_ppo_vs_mbam_test" \
--env "simple_tag" \
--train_mode 0 \
--test_mode 1 \
--seed -1 \
--prefix "train" \
--ranks 1 \
--device "cuda:0" \
--dir "" \
--eps_max_step 50 \
--eps_per_epoch 1 \
--save_per_epoch 100 \
--max_epoch 20000 \
--player2_is_ppo False \
--true_prob True \
#--prophetic_onehot True \
#--actor_rnn True \

#--max_epoch 100000 \


#--continue_train True \  #需要修改load文件