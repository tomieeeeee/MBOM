#!/bin/bash

# Begin experiment#
for _ in 1 2 3 4 5;do
  python3.6 main.py \
  --exp_name "simple_tag_ppo_0_vs_ppo_20000" \
  --env "simple_tag" \
  --prefix "test" \
  --test_mode 0 \
  --test_mp 5 \
  --seed -1 \
  --device "cuda:0" \
  --dir "" \
  --eps_max_step 50 \
  --eps_per_epoch 10 \
  --save_per_epoch 500 \
  --max_epoch 100 \
  --num_om_layers 2 \
  --true_prob True \
  --player2_is_ppo True \
  #--only_use_last_layer_IOP True \
  #--rnn_mixer True \
  #--actor_rnn True \
  #--policy_training True
  #--prophetic_onehot False \

    python3.6 main.py \
  --exp_name "simple_tag_ppo_1_vs_ppo_20000" \
  --env "simple_tag" \
  --prefix "test" \
  --test_mode 1 \
  --test_mp 5 \
  --seed -1 \
  --device "cuda:0" \
  --dir "" \
  --eps_max_step 50 \
  --eps_per_epoch 10 \
  --save_per_epoch 500 \
  --max_epoch 100 \
  --num_om_layers 2 \
  --true_prob True \
  --player2_is_ppo True \
  #--only_use_last_layer_IOP True \
  #--rnn_mixer True \
  #--actor_rnn True \
  #--policy_training True
  #--prophetic_onehot False \

  python3.6 main.py \
  --exp_name "simple_tag_mbam_1_vs_ppo_20000" \
  --env "simple_tag" \
  --prefix "test" \
  --test_mode 2 \
  --test_mp 5 \
  --seed -1 \
  --device "cuda:0" \
  --dir "" \
  --eps_max_step 50 \
  --eps_per_epoch 10 \
  --save_per_epoch 500 \
  --max_epoch 100 \
  --num_om_layers 2 \
  --true_prob True \
  --player2_is_ppo True \
  #--only_use_last_layer_IOP True \
  #--rnn_mixer True \
  #--actor_rnn True \
  #--policy_training True
  #--prophetic_onehot False \
done
