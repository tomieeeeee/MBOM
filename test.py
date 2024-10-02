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
    '''
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
    '''
    import numpy as np

    # 定义区域中心
    centers = [(j*0.2 + 0.1, i*0.5 + 0.25) for i in range(2) for j in range(5)]
    print(centers)
    # 假设有60个智能体的初始位置，这里随机生成为示例
    np.random.seed(0)  # 为了可复现性
    agent_positions = np.random.rand(60, 2)  # 生成60个智能体的x, y坐标
    print(agent_positions)
    # 计算每个智能体到各区域中心的距离，并进行分配
    def assign_agents_to_centers(agent_positions, centers):
        assignments = [[] for _ in centers]
        center_loads = [0] * len(centers)  # 跟踪每个中心已分配的智能体数量
 
        for agent_id, position in enumerate(agent_positions):
            # 计算到每个中心的距离并排序
            distances = sorted(enumerate([np.linalg.norm(position - np.array(center)) for center in centers]), key=lambda x: x[1])
            
            # 尝试分配到最近的中心，如果满了则尝试下一个
            for center_index, distance in distances:
                if center_loads[center_index] < 6:
                    assignments[center_index].append(agent_id)
                    center_loads[center_index] += 1
                    break

        return assignments

    # 分配智能体
    assignments = assign_agents_to_centers(agent_positions, centers)

    # 打印分配结果
    print(type(assignments))
    for i, group in enumerate(assignments):
        print(f"Area {i+1} ({centers[i]}): {group}")

    a= [[1,2],[3,4]]
    b= [[5,6],[8,7]]
    c= a+b
    print(c)
    for group in c:
        for num in group:
            print(num)
    summ = [29.690860369190816, -0.8341396308091864, -2.5029260356013783, -2.9716897090933205, -3.1853558472315653, -3.197896736978052, -3.2107127160393762, -4.504556925746452, -4.557067056426016, 10.273022084405593, -4.881647564415512, -4.882220868979319, -4.036000845133046, -3.8241509520425283, -3.8416795422506973, -3.557738261001618, -3.3464697353611443, -3.4887793321688436, -3.88748958361844, -4.220750388821731, 10.231873507685279, -5.215657956911969, -5.598466448339885, -5.903900950471055, -6.143863381641718, -6.721652932138854, -7.190382354216308, -7.56325791407543, -7.855550134421744, -8.082158724944962, -8.256394830144556, -8.908312235736702, -8.881605823512551, -8.861628569287367, -9.393722960407493, -10.420500301132552, -11.26358621107201, -11.940369741312551, -11.784315006308585, -11.691108181832568, -11.62233377455141, -11.57511559540778, -12.475983342669892, -13.952820892321963, -15.149386957334958, -16.100880165410686, -16.846739964091856, -17.425081066317677, -18.844038597394903, -18.165858060955923, -16.8918051364696, -16.023829058073677, -15.419140809724128, -14.990499674207989, -14.682544641316657, -15.353092016341282, -16.6850894612528, -17.589282675499433, -17.40413881879469, -18.008277392705075, -18.64741844319086, -19.14312566095326, -18.538916606289554, -18.262279039939795, -17.900990249288565, -16.933448885689803, -16.266206847368384, -16.583596070284507, -16.208967968131365, -15.920845491606475, -15.549314481724988, -16.02631270783309, -17.359195224494837, -19.386090980013623, -20.3544136113976, -21.243157967909003, -23.14895495390138, -23.806960280013598, -24.59349261043739, -25.255236208273327, -24.752490633098333, -23.497586905269696, -21.704180905297385, -20.39929662000601, -19.524865379530823, -17.91297215203894, -16.781203272476795, -16.300368933837518, -17.081778542641032, -16.45048402526678, -16.03566370998271, -16.176190293531864, -15.913927623701237, -16.536442947896337, -16.863276187679737, -17.273033929900826, -17.753926876421176, -18.13079774996833, -18.237187829469274, -18.144891834816196]
    total = sum(summ)
    #total = summ/1
    print(total)
    print(np.exp(20*(0.165-0.05)))
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

