def flatten_param(search_param):
    '''
    search_param format
    search_param = {
        "comb_param": [
            {
                "param_name": ["error_delay", "error_horizon", "mix_factor"],
                "param_value": [[0.9, 20, 10], [0.9, 10, 6]]
            },
        ],
        "param": {
            "opponent_model_learning_times": [1, 3, 5, 7, 10],
            "opponent_optimal_model_learning_rate": [0.005, 0.01, 0.05],
            "opponent_optimal_model_learning_times": [1, 3, 5, 7, 10],
            "v_learning_rate": [0.01],
            "a_learning_rate": [0.01],
        }
    }
    '''
    result = {}
    result_len = 0
    cur_result_len = 0
    comb_params = search_param["comb_param"]
    params = search_param["param"]
    for comb_param in comb_params:
        param_name = comb_param["param_name"]
        param_value = comb_param["param_value"]
        for name in param_name:
            if name not in result.keys():
                result[name] = []
        for value in param_value:
            for j in range(result_len):
                for key in result.keys():
                    if key not in param_name:
                        result[key].append(result[key][j])
                    else:
                        result[key].append(value[param_name.index(key)])
                    cur_result_len += 1
            if result_len == 0:
                for i, name in enumerate(param_name):
                    result[name].append(value[i])
                cur_result_len += 1
        result_len = len(result[param_name[0]]) if len(param_name) != 0 else 0
    for key in params.keys():
        if key not in result.keys():
            result[key] = []
        for i, v in enumerate(params[key]):
            for j in range(result_len):
                for k in result.keys():
                    if k == key:
                        result[k].append(v)
                    else:
                        if i == 0:
                            pass
                        else:
                            result[k].append(result[k][j])
            if result_len == 0:
                result[key].append(v)
        result_len = len(result[key])
    return result, result_len
