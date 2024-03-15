import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 3
        world.length_triangle = 0.5
        world.touch_range = 0.05
        world.sum_measure_min = 2 * np.math.sqrt(3) * world.length_triangle
        world.sum_measure_max = 4 * world.length_triangle
        world.measure_max = 2 * np.math.sqrt(3) * world.length_triangle / 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i] += 0.8
            landmark.index = i
        # set goal landmark
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.1, 0.1, 0.1])
            else:
                agent.color = np.array([0.9, 0.9, 0.9])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
            #agent.state.p_pos = np.array([0, world.measure_max / 2])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        init_pos = np.array([[-1 * world.length_triangle, 0], [1 * world.length_triangle,0], [0, np.sqrt(3) * world.length_triangle]])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = init_pos[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # the distance to the goal
        def landmark_score(agent, world):
            dis = np.array([np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) for landmark in world.landmarks])
            dis /= world.touch_range
            label = dis >= 1
            dis[dis < 1] = 1
            dis[label] = 0
            return dis, is_outside
        def compare_score(score1, score2):
            score1 = [int(s) for s in score1]
            score2 = [int(s) for s in score2]
            rew_matrix = [[0, 0, 0, 0],
                          [0.5, -1, 1, -1],
                          [0.5, -1, -1, 1],
                          [0.5, 1, -1, -1]]
            score_type = ["[0, 0, 0]", "[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"]
            score1 = score1.__str__()
            score2 = score2.__str__()
            s1_idx = -1
            s2_idx = -1
            for i, s in enumerate(score_type):
                if score1 == s: s1_idx = i
                if score2 == s: s2_idx = i
            #print("{},{}".format(s1_idx,s2_idx))
            return rew_matrix[s1_idx][s2_idx]


        score1, is_outside = landmark_score(agent, world)
        if is_outside:
            return -1.0
        score2 = 0
        for other in world.agents:
            if other is agent: continue
            else: score2, _ = landmark_score(other, world)
        #return np.sum(score1 - score2)
        return compare_score(score1, score2)

    def observation(self, agent, world):
        # 我的位置-landmark  对方的位置-landmark
        # get positions of all entities in this agent's reference frame
        agents_pos = [np.array([world.sum_measure_max]), np.array([world.measure_max])]
        for agt in world.agents:
            for entity in world.landmarks:
                agents_pos.append(entity.state.p_pos - agt.state.p_pos)
        return np.concatenate(agents_pos)