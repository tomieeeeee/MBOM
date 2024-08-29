from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import os
import numpy
def idx_to_dis_idx(idx):
    "71 -> 2 4 1"
    a = numpy.floor(idx / 25).astype(numpy.int).item()
    b = numpy.floor((idx - a * 25) / 5).astype(numpy.int).item()
    c = idx - a * 25 - b * 5
    return [a, b, c]
if __name__ == "__main__":
    #graphviz = GraphvizOutput()
    #graphviz.output_file = 'basic.png'
    #with PyCallGraph(output=graphviz): 
    #    print("HELLO")
    #cmd = 'dot -Tpng ex.dot -obbasic.png D:/document/MBAM/tmpoaslx825'
    #ret = os.system(cmd)
    actionss = numpy.random.randint(0, 5, size=[3])
    oop = numpy.random.randint(0, 5, size=[3])
    SIMPLE_TAG_n_action = 5#[5, 5, 5]
    SIMPLE_TAG_a_onehot = numpy.eye(SIMPLE_TAG_n_action)
    action = SIMPLE_TAG_a_onehot[actionss[1]]
    print(action)
    for idx in actionss:
        b = SIMPLE_TAG_a_onehot[idx].copy() +action
        print(b)
    oppo = idx_to_dis_idx(20)
    print(oppo)
    batchsize = 1
    state = numpy.random.random((batchsize, 20))

    actions = ([numpy.random.randint(0, 5, 3) for i in range(batchsize)],numpy.random.randint(0, 5, batchsize))
    
    list1 = [[ numpy.random.randint(0, 5, size=[1]),numpy.random.randint(0, 5, size=[1]),numpy.random.randint(0, 5, size=[1])],numpy.random.randint(0, 5, size=[1])]
    list = [[action.item() for  action in list1[0]],list1[1].item()]
    
    print(state)
    print(list)
    print(type(list))
    #action = SIMPLE_TAG_a_onehot(actions)
    arr = numpy.array([1, 2, 3, 4, 5])

# 转换为元组
    tup = tuple(arr)
    print(tup)
    list=[0,1,2,3]
    list1=[4,5,6]
    list.append(list[0:(2)])
    print(list)
    n_action=[]
    for i in range(50):
        n_action.append(5)
    print(n_action)
    import torch
    import torch.nn as nn
    state = numpy.random.random((11, 20))
    #layer = layer.reshape(-1,2)
    print(state)
    a=[1,2,3,4,5,6,7,8,9,10,11,12]
    a1=a[0:6]
    a2=a[6:]
    a=[a1,a2]
    print(a1)
    print(a2)
    print(a)
    print(13//2)
    a=[1,2,3,4,5,6,7,8,9,10,11,12]
    reward1 =reward2=0
    for i in range(len(a[0:6])):
            reward1 += a[0:6][i]###########根据智能体数量要变化
    for j in range(len(a[6:])):
            reward2 += a[6:][j]
        
    print(reward1,reward2)
    import torch


# 随机生成一个尺寸为 [2, 6, 5] 的数据
    oppo_hidden_prob = torch.randn(2, 6, 5)

# 打印原始数据的尺寸
    print("Original size:", oppo_hidden_prob)

# 转换数据的尺寸从 [2, 6, 5] 到 [6, 2, 5]
    oppo_hidden_prob_reshaped = oppo_hidden_prob.transpose(0, 1)

# 打印转换后数据的尺寸
    print("Reshaped size:", oppo_hidden_prob_reshaped.shape)

# 打印转换后数据以验证
    print("Reshaped data:\n", oppo_hidden_prob_reshaped)
    chr="0x"+"2192"
    print(chr)
    def t():
         return -10
    a= 10
    a+=t()
    print(a)
    t= [1,2,3]
    b=[4,5,6]
    print(t+b)

    '''v6
    def is_collision(self, agent1, agent2,world,hit_list):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        agent_index = world.agents.index(agent2)
        if hit_list[agent_index]:
            return True if dist < agent2.combat_radius else False
        else:
            return False


    def reward(self, agent, world,hit_list):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world,hit_list) if agent.adversary else self.agent_reward(agent, world,hit_list)
        return main_reward

    def agent_reward(self, agent, world,hit_list):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        agents = self.good_agents(world)
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
              for ag in agents:
                if self.is_collision(a, ag,world,hit_list):
                    #print("good hit")
                    rew += 10

        '''# agents are penalized for exiting the screen, so that they can be caught by the adversaries
        #x/y坐标
        #[0,0.9]不惩罚
        #[0.9,1]超出0.9的部分*10
        #[1,无穷]指数惩罚
    '''
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)#e^(2x-2)和10的较小值
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world,hit_list):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:######################只要碰到所有智能体都得分
                    if self.is_collision(ag, adv,world,hit_list):
                        #print("bad hit")
                        rew += 10
    
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)#e^(2x-2)和10的较小值
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew
    '''
'''environment
    hit_list = []
        print("OKKKK")
        if world.num_agents = 12:
            print("OKKKK")
            for i, agent in enumerate(world.agents):
               hit_list[i] = random_true(agent.hit(hit_probability))
               return self.reward_callback(agent, self.world,hit_list)
'''
'''
    
    array_example = numpy.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=numpy.float32)
    list_with_array = []
    for i in range(6):
       list_with_array.append(array_example)
    
    
    total = []
    for i in range(3):
        total.append(list_with_array)
    #after = numpy.stack(total)
    to_dict = []
    for i in range(len(total)):
        temp=total[i]
        temp_list=[]
        for j in range(len(temp)):
              print(numpy.stack(temp[j]))
              np = numpy.stack(temp[j]).squeeze(axis=1)
              temp_list.append(np)
        to_dict.append(temp_list)
    #to_dict = after.squeeze(axis=1)

# 打印这个列表，以展示包含 NumPy 数组的完整结构
    print(list_with_array)
    
    print(to_dict)

'''

