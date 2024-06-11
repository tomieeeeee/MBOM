#gfootball academy_run_to_score_with_keeper 1v1
'''
    import gfootball
    from env_model.football_1_vs_1_penalty_kick import Env_Model
    env = Env_Model(env=gfootball.env.create_environment(env_name="tests.1_vs_1_penalty_kick",
                                                         # "academy_run_to_score_with_keeper",
                                                         representation="simple115",
                                                         rewards='scoring',  # 'scoring,checkpoints',
                                                         number_of_left_players_agent_controls=1,
                                                         number_of_right_players_agent_controls=1,
                                                         # stacked=False,
                                                         # logdir='./log/',
                                                         # write_goal_dumps=True,
                                                         # write_full_episode_dumps=False,
                                                         # write_video=False,
                                                         # dump_frequency=1000,
                                                         ))
'''
player1_conf = {
    "conf_id": "player1_conf",
    # env setting
    "n_state": 52,
    "n_action": [5, 5, 5,5,5,5],
    "n_opponent_action": [5, 5, 5,5,5,5],
    "action_dim": 6,
    "type_action": "discrete",  # "discrete", "continuous"
    "action_bounding": 0,  # [()]
    "action_scaling": [1, 1],
    "action_offset": [0, 0],
    

    #shooter ppo setting
    "v_hidden_layers": [128, 64],
    "a_hidden_layers": [128, 64],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,  # value discount factor
    "lambda": 0.99,  # general advantage estimator
    "epsilon": 0.3,  # ppo clip param#越大越探索
    "entcoeff": 0.01, 
    "a_update_times": 10,
    "v_update_times": 10,
    "buffer_memory_size": 6000, #eps_per_epoch * save_per_epoch = 10 * 30

    # opponent model setting
    "num_om_layers": 1,
    "opponent_model_hidden_layers": [128, 64],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,
}
player2_conf = {
    "conf_id": "player2_conf",
    # env setting
    "n_state": 52,
    "n_action": [5, 5, 5,5,5,5],
    "n_opponent_action": [5, 5, 5,5,5,5],
    "action_dim": 6,
    "type_action": "discrete",  # "discrete", "continuous"
    "action_bounding": 0,  # [()]
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    # opponent model setting
    "num_om_layers": 2,
    "opponent_model_hidden_layers": [128, 64],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,

    "imagine_model_learning_rate": 0.001,
    "imagine_model_learning_times": 5,  # times is 0 means training until optimal action is max prob , but unfinished
    "roll_out_length": 1,
    "short_term_decay": 0.9,  # error discount factor
    "short_term_horizon": 10,
    "mix_factor": 1,  # adjust mix cuvre, Strongly related to error_horizon and error_delay

    # ppo setting
    "v_hidden_layers": [128, 64],
    "a_hidden_layers": [128, 64],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,  # value discount factor
    "lambda": 0.99,  # general advantage estimator
    "epsilon": 0.3,  # ppo clip param#越大越探索
    "entcoeff": 0.01, 
    "a_update_times": 10,
    "v_update_times": 10,
    # ppo buffer setting
    "buffer_memory_size": 6000, #update_episode * max_episode_step = 20 * 30
}

player1_reversed_conf = {
    "conf_id": "player1_conf",
    # env setting
    "n_state": 52,
    "n_action": [5, 5, 5,5,5,5],
    "n_opponent_action": [5, 5, 5,5,5,5],
    "action_dim":6,
    "type_action": "discrete",  # "discrete", "continuous"
    "action_bounding": 0,  # [()]
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    #shooter ppo setting
    "v_hidden_layers":[128, 64],
    "a_hidden_layers":[128, 64],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,  # value discount factor
    "lambda": 0.99,  # general advantage estimator
    "epsilon": 0.115,  # ppo clip param
    "entcoeff": 0.0015,  #0.0015
    "a_update_times": 10,
    "v_update_times": 10,
    "buffer_memory_size": 6000, #eps_per_epoch * save_per_epoch = 10 * 30

    # opponent model setting
    "num_om_layers": 1,
    "opponent_model_hidden_layers":[128, 64],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,
}
player2_reversed_conf = {
    "conf_id": "player2_conf",
    # env setting
    "n_state": 52,
    "n_action":[5, 5, 5,5,5,5],
    "n_opponent_action": [5, 5, 5,5,5,5],
    "action_dim": 6,
    "type_action": "discrete",  # "discrete", "continuous"
    "action_bounding": 0,  # [()]
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    # opponent model setting
    "num_om_layers": 2,
    "opponent_model_hidden_layers": [128, 64],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,

    "imagine_model_learning_rate": 0.001,
    "imagine_model_learning_times": 5,  # times is 0 means training until optimal action is max prob , but unfinished
    "roll_out_length": 1,
    "short_term_decay": 0.9,  # error discount factor
    "short_term_horizon": 10,
    "mix_factor": 1,  # adjust mix cuvre, Strongly related to error_horizon and error_delay

    # ppo setting
    "v_hidden_layers":[128, 64],
    "a_hidden_layers": [128, 64],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,  # value discount factor
    "lambda": 0.99,  # general advantage estimator
    "epsilon": 0.115,  # ppo clip param
    "entcoeff": 0.0015,
    "a_update_times": 10,
    "v_update_times": 10,
    # ppo buffer setting
    "buffer_memory_size": 6000, #update_episode * max_episode_step = 20 * 30
}