import torch
from torch.nn.parameter import Parameter
import types

def soft_update(params, ratio):
    '''
    :param params: list [param0, param1, ...]
    :param ratio:  list [ratio0, ratio1, ...]   sum must is 1.0
    :return: param0 * ratio0 + param1 * ratio1 + ......
    '''
    assert abs(sum(ratio) - 1.0) < 1e-5, "soft_update ratio sum is not 1.0"
    assert len(params) == len(ratio), "soft_update param length error"
    if isinstance(params[0], types.GeneratorType):  #model.parameters is generator type
        params = [list(p) for p in params]

    target = [Parameter(torch.zeros_like(p)) for p in params[0]]
    for j in range(len(params[0])):
        for i in range(0, len(ratio)):
            target[j].data.copy_(target[j].data + params[i][j] * ratio[i])
    return target