#!/bin/bash
#  ppo_mh  N vs 1 mbam_om_mh
# Begin experiment#
python main.py \
--exp_name "PPO_test_1V3" \
--env "simple_tag" \
--train_mode 0 \
--test_mode 1 \
--policy_training True \
--player2_is_ppo True \
--seed -1 \
--prefix "test" \
--ranks 1 \
--device "cuda:0" \
--dir "" \
--ranks 1 \
--eps_max_step 100 \
--eps_per_epoch 1 \
--save_per_epoch 100 \
--max_epoch 200000 \
--true_prob True \