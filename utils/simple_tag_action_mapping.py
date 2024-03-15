import numpy as np

'''
SIMPLE_TAG_n_action = [3, 4, 5]
temp = np.arange(SIMPLE_TAG_n_action[0]).reshape(-1,1)
for i in range(1, len(SIMPLE_TAG_n_action)):
    temp = np.concatenate([np.repeat(temp, SIMPLE_TAG_n_action[i], axis=0), np.tile(np.arange(SIMPLE_TAG_n_action[i]).reshape(-1,1), [temp.shape[0], 1])], axis=1)
'''




SIMPLE_TAG_n_action = 5
SIMPLE_TAG_a_onehot = np.eye(SIMPLE_TAG_n_action)
SIMPLE_TAG_a_onehot_2 = np.concatenate([np.repeat(SIMPLE_TAG_a_onehot, SIMPLE_TAG_n_action, axis=0), np.tile(SIMPLE_TAG_a_onehot, [SIMPLE_TAG_n_action, 1])], axis=1)
SIMPLE_TAG_a_onehot_3 = np.concatenate([np.repeat(SIMPLE_TAG_a_onehot_2, SIMPLE_TAG_n_action, axis=0), np.tile(SIMPLE_TAG_a_onehot, [SIMPLE_TAG_n_action * SIMPLE_TAG_n_action, 1])], axis=1)

# idx 71
# dis_idx = 2 4 1
# onehot =  001001000000010
# dis_onehot =  00100 10000 00010

def idx_to_onehot(idx):
    "125 -> 000010010000100"
    return SIMPLE_TAG_a_onehot_3[idx].copy()

def onehot_to_idx(onehot):
    "000010010000100 -> 125"
    return np.argmax(np.all(SIMPLE_TAG_a_onehot_3 - onehot == 0, axis=1)).item()

def dis_idx_to_idx(dis_idx):
    "2 4 1 -> 71"
    return dis_idx[0] * SIMPLE_TAG_n_action * SIMPLE_TAG_n_action + dis_idx[1] * SIMPLE_TAG_n_action + dis_idx[2]

def idx_to_dis_idx(idx):
    "71 -> 2 4 1"
    a = np.floor(idx / 25).astype(np.int).item()
    b = np.floor((idx - a * 25) / 5).astype(np.int).item()
    c = idx - a * 25 - b * 5
    return [a, b, c]

def dis_onehot_to_dis_idx(dis_onehot):
    return [np.argwhere(onehot).item() for onehot in dis_onehot]

def dis_idx_to_dis_onehot(dis_idx):
    return [SIMPLE_TAG_a_onehot[idx].copy() for idx in dis_idx]

def onehot_to_dis_onehot(onehot):
    return np.split(onehot, 3)

def dis_onehot_to_onehot(dis_onehot):
    return np.concatenate(dis_onehot)