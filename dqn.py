#!/usr/bin/env python
# coding: utf-8

# # 强化学习

# 强化学习是机器学习大家族中的一大类，使用强化学习能够让机器学着如何在环境中拿到高分，表现出优秀的成绩。
# 
# 而这些成绩背后却是他所付出的辛苦劳动，不断的试错，不断地尝试，累积经验，学习经验。

# <table><tr><td><img src='./images/alphago.jpeg' style=width:600px;></td><td><img src='./images/starcraft.jpg' style=width:600px;></td></tr></table>

# 实际中的强化学习例子有很多，比如有名的AlphaGo，机器第一次在围棋场上战胜人类高手；再比如让计算机学会玩即时策略游戏星际争霸等。
# 
# 这些都是让计算机在不断的尝试中更新自己的行为准则，从而一步步学会如何下好围棋，如何操控游戏得到高分。
# 
# 既然要让计算机自己学，那计算机通过什么来学习呢？

# <img src='./images/teacher.png' style=width:600px;>

# 计算机也需要一位虚拟的老师，这个老师比较吝啬，他不会告诉你如何移动，如何做决定，他为你做的事只有给你的行为打分。
# 
# 那我们应该以什么形式学习这些现有的资源，或者说怎么样只从分数中学习到我应该怎样做决定呢？
# 
# 很简单，我们只需要记住那些高分，低分对应的行为，下次用同样的行为拿高分，并避免低分的行为。

# <img src='./images/vs_supervised.png' style=width:600px;>

# **对比监督学习**
# 
# 监督学习，是已经有了数据和数据对应的正确标签。比如上图中，监督学习算法就能学习出那些脸对应哪种标签。
# 
# 不过强化学习还要更进一步，一开始它并没有数据和标签。
# 
# 它要通过一次次在环境中的尝试，获取这些数据和标签，然后再学习通过哪些数据能够对应哪些标签，通过学习到的这些规律，就有可能地选择带来高分的行为。

# **核心要素**

# 强化学习中有状态（state）、动作（action）、奖赏（reward）这三个要素。
# 
# 智能体（Agent）会根据当前状态来采取动作，并记录被反馈的奖赏，以便下次再到相同状态时能采取更优的动作。
# 
# 我们接下来以一款曾经很火的手机游戏`Flappy Bird`来举例来说明，你需要控制小鸟不撞到随机出现的水管。

# **状态选择**

# 最直观的状态提取方法就是以游戏每一帧的画面为状态。
# 
# 但为了简化问题，取小鸟到下一组管子的水平距离和垂直距离差作为小鸟的状态。更准确地说，$\Delta x$与$\Delta y$的定义如下图所示

# <p align=center><img src="./images/flappybird.png" alt="flappybird" style="width:400px;"/></p>

# **动作选择**

# 每一帧，小鸟只有两种动作可选：1.向上飞一下。2.什么都不做

# **奖赏定义**

# 小鸟活着时，每一帧给予1的奖赏；若死亡，则给予-1000的奖赏

# # Q Learning

# ## 算法介绍

# 提到Q-learning，我们需要先了解Q的含义
# 
# Q为动作效用函数（action-utility function），用于评价在特定状态下采取某个动作的优劣。可以看作是智能体（Agent）的记忆

# 在这个问题中，状态和动作的组合是有限的。
# 
# 所以我们可以把Q当做是一张表格。表中的每一行记录了状态（$\Delta x$，$\Delta y$）和选择不同动作（飞或不飞）时的奖赏

# 状态                                | 飞      | 不飞
# ---------------------------------- | :-----: | :-----
# <img width=100/>                   |         |
# \($\Delta x_1$, $\Delta y_1$\)     |  1      |  20
# \($\Delta x_2$, $\Delta y_2$\)     |  20     |  -100
#             ...                    | ...     | ...
# \($\Delta x_m$, $\Delta y_{n-1}$\) |  -100   |  2
# \($\Delta x_m$, $\Delta y_n$\)     |  50     |  -200

# 理想状态下，在完成训练后，我们会获得一张完美的Q表格。
# 
# 我们希望只要小鸟根据当前位置查找到对应的行，选择效用值较大的动作作为当前帧的动作，就可以无限地存活。

# **伪代码**

# 初始化表格Q
# 
# while Q 未收敛：
# 
#     初始化小鸟的位置S，开始新一轮的游戏
#     
#     while S != 死亡状态:
#     
#         使用策略π，获得动作a=π(S)
#         
#         使用动作a进行游戏，获得小鸟最新的位置S'与奖励R(S)
#         
#         Q[S,A] ← (1-α)*Q[S,A] + α*(R(S,a) + γ* max Q[S',a])
#         
#         S ← S'

# 为避免使Q陷入局部最优，策略π中引入了贪婪度的概念，每个状态以一定概率选择飞或不飞，剩下的概率选择随机动作

# Q表格将根据以下公式进行更新：

# <img src='./images/qlearning.svg' style=width:500px;>

# 其中，$\alpha$为学习率（learning rate），$\gamma$为奖励衰减因子
# 
# 学习率$\alpha$越大，保留之前训练的结果就越少
# 
# 奖励衰减因子$\gamma$越大，越重视以往的经验，越小的话，小鸟会变得只重视眼前的利益R

# ## 示例1——寻找宝藏

# 这是使用表格Q Learning方法进行强化学习的一个简单示例

# 例子的环境是一个一维世界，在世界的右边有宝藏，勇士只要得到宝藏尝到了甜头，然后以后就记住了得到宝藏的方法
# 
# 勇士Agent在一维世界的左边，宝藏在最右边的位置，如下面序列所示
# 
# O - - - - - - T

# 我们首先导入一些数据处理和展示结果用的包
# 
# 为了保证每次结果的一致，我们设置随机种子，让计算机产生一组伪随机数

# In[1]:


import numpy as np
import pandas as pd
import time

np.random.seed(2)  # 为了方便复现结果


# 一些参数配置

# In[2]:


N_STATES = 6                    # 这个一维世界的长度
ACTIONS = ['left', 'right']     # 可选的行为
EPSILON = 0.9                   # 贪婪度，多少几率选择最优行为
ALPHA = 0.1                     # 学习率
GAMMA = 0.9                     # 对未来奖励的衰减因子
MAX_EPISODES = 10               # 游戏的回合数
FRESH_TIME = 0.3                # 每一次移动的时间间隔，为了方便看清agent的移动情况，可以调大刷新时间


# **Q表**

# 构建一个M * N的Q表格，用于记录不同状态对不同行为的值，其中M为状态个数，N为行动的个数
# 
# 更新Q表也就是在更新勇士的行为准则。Q表的行对应所有的state（勇士的位置），列对应action（勇士的行动）

# In[3]:


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q表的初始值均为0
        columns=actions,                     # 行为的名字
    )
    print(table)
    return table


# **定义动作**

# 现在我们来定义勇士如何挑选行为。这里我们引入贪婪度的概念。
# 
# 因为在初始阶段，随机的探索环境，往往比固定的行为模式要好，所以在到达新地点时，正是累积经验的阶段，我们希望勇士不会那么贪婪。
# 
# 随着探索世界的不断提升，勇士知道了往哪边走是更接近宝藏的，也就变得越来越贪婪，选择最优的行动。
# 
# 在这个示例中，我们设置勇士的贪婪度为0.9，也就是说90%的时间选择最优策略，10%的时间用来试错

# In[4]:


# 在某个state地点, 选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个地点的所有行为值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 如果大于贪婪度或，非贪婪或者这个地点还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # 贪婪模式
    return action_name


# **环境反馈**

# 做出行为后，环境也要给我们的行为一个反馈, 反馈出在下个地点S_和在上地点S做出行为A所得到的奖励R。
# 
# 这里定义的规则就是, 只有当勇士移动到了终点T，勇士才会得到唯一的一个奖励，奖励值R=1，其他情况都没有奖励。

# In[5]:


def get_env_feedback(S, A):
    # 如下代码定义勇士如何与环境产生互动
    if A == 'right':  # 向右移动
        if S == N_STATES - 2:  # 到达终点
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # 向左移动，始终到达不到终点，奖励值始终为0
        R = 0
        if S == 0:
            S_ = S  # 回到起点
        else:
            S_ = S - 1
    return S_, R


# **环境更新**

# In[6]:


def update_env(S, episode, step_counter):
    # 如下定义环境如何被更新
    env_list = ['-'] * (N_STATES - 1) + ['T']   # '---------T'表示我们的环境
    if S == 'terminal':
        interaction = f'Episode {episode + 1}: total_steps = {step_counter}'
        print(f'\r{interaction}', end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'O' # 表示勇士处在哪一个地点
        interaction = ''.join(env_list)
        print(f'\r{interaction}', end='')
        time.sleep(FRESH_TIME)


# **主循环**

# Q Learning伪代码实现

# In[7]:


def run():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0  # 记录当前回合勇士走了多少步
        S = 0  # 表示勇士从起点出发
        is_terminated = False  # 表示勇士是否到达终点
        update_env(S, episode, step_counter)  # 更新环境，展示勇士的位置

        # 当勇士没有到达终点，该回合不会结束，直至勇士到达终点
        while not is_terminated:
            A = choose_action(S, q_table)  # 基于当前的state和Q表选择下一步的行为
            S_, R = get_env_feedback(S, A)  # 采取行为并得到环境的反馈
            q_predict = q_table.loc[S, A]  # 估算的（状态-行为）的值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # 实际的（状态 - 行为）的值，回合未结束
            else:
                q_target = R  # 实际的（状态 - 行为）值，得到奖励，回合结束
                is_terminated = True  # 更新为已到达终点的状态

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # Q表更新，更新在状态S采取A行为的值
            S = S_  # 移动到下一个地点state

            update_env(S, episode, step_counter + 1)  # 更新环境，展示勇士的位置
            step_counter += 1  # 勇士步数+1
    return q_table


# In[8]:


q_table = run()
print("\r\nQ-table:\n")
print(q_table)
# 4的位置向右走价值最大，最接近终点。


# ## 示例2——走迷宫1.0

# 现在我们来看一个更加复杂一点的例子。让勇士学会走迷宫，黄色代表宝藏，黑色代表深渊陷阱。勇士通过不断尝试，学会避让陷阱，最终找到宝藏。

# <p align=center><img src="./images/maze.png" alt="maze" style="width:400px;"/></p>

# In[9]:


import time

import numpy as np
import pandas as pd
import tkinter as tk


# 首先利用tkinter构建一个迷宫世界

# tkinter是python自带的简单GUI模块，通过tkinter模拟的环境，我们可以清楚的观察到勇士的行动路径

# In[10]:


UNIT = 40   # 每个迷宫单元格的宽度
MAZE_H = 4  # 迷宫的高度
MAZE_W = 4  # 迷宫的宽度


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ["up", "down", "left", "right"] # 定义动作列表，有四种动作，分别为上下左右
        self.n_actions = len(self.action_space)
        self.title("Maze")
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT)) # 设置迷宫大小
        self._build_maze()

    def _build_maze(self):
        """构建迷宫
        """
        # 设置迷宫界面的背景
        self.canvas = tk.Canvas(self, bg="white",
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 划分迷宫单元格，即根据坐标位置划线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 起点位置
        origin = np.array([20, 20])

        # 创建陷阱1
        trap1_center = origin + np.array([UNIT * 2, UNIT])
        self.trap1 = self.canvas.create_oval(
            trap1_center[0] - 15, trap1_center[1] - 15,
            trap1_center[0] + 15, trap1_center[1] + 15,
            fill='black')

        # 创建陷阱2
        trap2_center = origin + np.array([UNIT, UNIT * 2])
        self.trap2 = self.canvas.create_oval(
            trap2_center[0] - 15, trap2_center[1] - 15,
            trap2_center[0] + 15, trap2_center[1] + 15,
            fill='black')

        # 创建宝藏
        treasure_center = origin + UNIT * 2
        self.treasure = self.canvas.create_rectangle(
            treasure_center[0] - 15, treasure_center[1] - 15,
            treasure_center[0] + 15, treasure_center[1] + 15,
            fill='yellow')

        # 创建可以移动的红色格子代表勇士，并放置在起始位置
        self.warrior = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 组合所有元素
        self.canvas.pack()

    def reset(self):
        """重置迷宫界面
        """
        self.update()  # 更新tkinter的配置
        time.sleep(0.5)

        # 删除当前勇士的位置，重置其回到起点
        self.canvas.delete(self.warrior)
        origin = np.array([20, 20])
        self.warrior = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 返回勇士的起始位置
        return self.canvas.coords(self.warrior)

    def step(self, action):
        """根据动作，更新迷宫状态
        """
        state = self.canvas.coords(self.warrior)

        base_action = np.array([0, 0])
        if action == 0:     # 向上
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 向下
            if state[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 向左
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # 向右
            if state[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        # 移动勇士位置
        self.canvas.move(self.warrior, base_action[0], base_action[1])

        # 移动后勇士的位置
        state_next = self.canvas.coords(self.warrior)

        # 奖励函数
        # 到达宝藏位置奖励1，到达陷阱处奖励-1，其他位置奖励0
        if state_next == self.canvas.coords(self.treasure):
            reward = 1
            done = True
            state_next = 'terminal'
        elif state_next in [self.canvas.coords(self.trap1), self.canvas.coords(self.trap2)]:
            reward = -1
            done = True
            state_next = 'terminal'
        else:
            reward = 0
            done = False

        return state_next, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


# 接下来，我们需要定义一个Q Learing的类，在其中定义Q Learning的计算过程。主要的成员函数有：
# 
# 1. choose_action：根据所在的state, 或者是在这个state上的 观测值来决策
# 2. learn：与上一个例子一样，我们根据terminal state终止符来判断是否继续更新q表格。更新的方式与神经网络类似，学习率 * (真实值 - 预测值)，然后将判断误差传递回去更新Q表上对应的值
# 3. check_state_exist：如果还没有当前state, 那我们就插入一组全0数据, 当做这个state的所有行为的初始值。这里没有像上一个示例那样，直接生成整个Q表，因为有些迷宫位置永远不会到达，比如本示例中的最右下角。

# In[11]:


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon_greedy=0.9):
        self.actions = actions  # 行为列表
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = epsilon_greedy  # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions)  # 初始化q表格

    def choose_action(self, observation):
        """选择行为"""
        self.check_state_exist(observation)  # 检查当前state在q表格中是否存在

        if np.random.uniform() < self.epsilon:  # 选择q值最高的行为
            state_action = self.q_table.loc[observation, :]
            # 同一个state，可能会出现多个相同的Q行为值，我们选取其中的任一一个
            action_cands = state_action[state_action == np.max(state_action)].index
            action = np.random.choice(action_cands)
        else:  # 随机选择行为
            action = np.random.choice(self.actions)

        return action

    def learn(self, state, action, reward, state_next):
        """学习更新参数"""
        self.check_state_exist(state_next)

        q_predict = self.q_table.loc[state, action]
        if state_next != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[state, :].max()  # 当state不是终止符时，取q表中对应最大价值的行动
        else:
            q_target = reward  # 下一个state是终止符，获取其奖励，可能是1或者-1

        # 更新对应的state-action值
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        """检查state是否存在"""
        # 表格中的index就是每个迷宫位置的坐标
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )


# 最后是Q Learing的主流程代码，是整个Q learning最重要的迭代更新部分
# 
# 挑选动作 -> 获取环境反馈 -> 学习误差更新Q表 -> 直至达到终止条件

# In[12]:


def update(q, env, num_episode=100):
    num_fails = 0  # 记录失败的次数
    
    for i in range(num_episode):

        # 初始化state的观测值
        # 观测值为state的坐标，如(5,5,35,35)
        observation = env.reset()
        num_steps = 0

        while True:
            # 重置可视化环境
            env.render()

            # 基于观测值挑选动作
            # 这里我们将坐标转为字符串代表该state，不再进行多一层的映射了
            action = q.choose_action(str(observation))

            # 采取行动并从环境中得到下一个观测值和奖励
            observation_next, reward, done = env.step(action)

            # 从这次学习更新Q表对应state的值
            q.learn(str(observation), action, reward, str(observation_next))

            # 观测值交换
            observation = observation_next

            # 累计一次尝试次数
            num_steps += 1

            # 当获取到终止条件后，结束本回合
            if done:
                find_treasure = "Yes" if reward == 1 else "No"
                if find_treasure == "No":
                    num_fails += 1
                print(f"Episode {i}: use {num_steps} steps.\tFind treasure: {find_treasure}")
                break
    
    print(f"Failed {num_fails}/{num_episode} times")


# 实例化环境和Q表
# 
# 需要注意的是notebook里执行可能会出现错误，请在cmd窗口执行maze.py

# In[13]:


# 初始化环境实例
env = Maze()

# 初始化Q表实例，传入四种动作的id列表
q = QLearningTable(actions=list(range(env.n_actions)))

# 主窗口将会在100毫秒启动update，执行走迷宫
env.after(100, update(q, env))
env.mainloop()


# 前10回合，只成功了两次

# Episode 0: use 10 steps.	Find treasure: No
# 
# Episode 1: use 14 steps.	Find treasure: No
# 
# Episode 2: use 26 steps.	Find treasure: No
# 
# Episode 3: use 28 steps.	Find treasure: No
# 
# Episode 4: use 11 steps.	Find treasure: No
# 
# Episode 5: use 186 steps.	Find treasure: No
# 
# Episode 6: use 29 steps.	Find treasure: No
# 
# Episode 7: use 71 steps.	Find treasure: Yes
# 
# Episode 8: use 10 steps.	Find treasure: No
# 
# Episode 9: use 17 steps.	Find treasure: Yes
# 
# Episode 10: use 8 steps.	Find treasure: No

# 而到了最后10回合，已经可以成功7次了

# Episode 90: use 28 steps.	Find treasure: No
# 
# Episode 91: use 19 steps.	Find treasure: Yes
# 
# Episode 92: use 102 steps.	Find treasure: Yes
# 
# Episode 93: use 18 steps.	Find treasure: Yes
# 
# Episode 94: use 27 steps.	Find treasure: Yes
# 
# Episode 95: use 23 steps.	Find treasure: No
# 
# Episode 96: use 93 steps.	Find treasure: No
# 
# Episode 97: use 33 steps.	Find treasure: Yes
# 
# Episode 98: use 47 steps.	Find treasure: Yes
# 
# Episode 99: use 39 steps.	Find treasure: Yes

# # Deep Q-Network

# ## 算法介绍

# DQN是一种融合了神经网络和Q learning的方法, 名字叫做Deep Q Network。
# 
# 谷歌的DeepMind团队就是靠着这DQN网络使计算机玩电动玩得比我们人还厉害

# **Q Learing局限**

# 在Q Learning中，我们使用表格存储每一个状态state以及和这个state对应的行为action的Q值。
# 
# 但当问题无比复杂，需要存储的状态无穷无尽（比如下围棋），如果全部用Q表来存储它们，计算机内存可能会不够而且搜索状态也变得非常耗时

# <p align=center><img src="./images/dqn.png" alt="dqn" style="width:600px;"/></p>

# 这时候，神经网络前来报道，它很擅长端到端的计算。我们把状态作为输入给到神经网络，它根据网络中的记忆输出每一种动作的Q值，然后选择拥有最大值的动作当做下一步要做的动作。
# 
# 我们可以想象成，神经网络接受外部的信息，相当于眼睛鼻子耳朵收集信息，然后通过大脑加工输出每种动作的值，最后通过强化学习的方式选择动作。

# **训练技巧**

# 1. 经验回顾：主要解决样本关联性和利用效率的问题。使用一个记忆池存储多条经验[s,a,r,s’]，再从中随机抽取一批数据送去训练
# 
# 2. 固定Q目标：主要解决算法训练不稳定的问题。复制一个和原来Q网络结构一样的Target Q网络，用于计算Q目标值

# **网络结构**

# 使用两个神经网络是为了固定住一个神经网络 (target_net) 的参数。
# 
# target_net是predict_net的一个历史版本, 拥有predict_net很久之前的一组参数, 而且这组参数会被固定一段时间, 然后再被predict_net的新参数所替换。
# 
# 而predict_net是不断在被提升的, 所以是一个可以被训练的网络。

# <p align=center><img src="./images/dqn2.png" alt="dqn2" style="width:600px;"/></p>

# ## 示例1——走迷宫2.0

# In[3]:


import collections
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras


# 本示例我们还是基于熟悉的迷宫环境，重点在于实现DQN算法，之后我们再拿着做好的DQN算法去跑其他更有意思的环境
# 
# DQN算法与Q Learning的主框架类似，在其上加了一些装饰：
# 
# 1. 记忆池，用于重复学习
# 2. 神经网络计算Q值
# 3. 暂时冻结预测网络的参数，切断相关性

# 与Q表直接返回是否为terminal不同，DQN只能接受数字输入，我们这里重新定义特征
# 
# 由于，我们将勇士放置在左上角，勇士只能向下或向右走才能到达宝藏位置，于是我们将当前勇士的位置到宝藏距离作为新的特征

# <p align=center><img src="./images/maze2.png" alt="maze2" style="width:400px;"/></p>

# 基于value的强化学习的优化目标一般为：$\min \left\|Q\left(s_{t}, a_{t}\right)-\left(r_{t}+Q\left(s_{t+1}, a_{t+1}\right)\right)\right\|_{2}$
# 
# 上式的目的是希望学到一个尽可能准确的Q函数。训练阶段，训练集可以看做是一个个的$(s,a,r,s_{+1})$元组，而上式的是根据当前的Q函数，输入状态$s_{t+1}$以及所有待选动作，最后选出来的Q值最大的动作
# 
# 我们通过MSE损失来刻画这个误差，希望实际的奖励和估计出来的奖励之间的差距越来越小

# In[4]:


class Maze2(tk.Tk, object):
    def __init__(self, n_features):
        super(Maze2, self).__init__()
        self.action_space = ["up", "down", "left", "right"] # 定义动作列表，有四种动作，分别为上下左右
        self.n_actions = len(self.action_space)
        self.n_features = n_features
        self.title("Maze")
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT)) # 设置迷宫大小
        self._build_maze()

    def _build_maze(self):
        """构建迷宫
        """
        # 设置迷宫界面的背景
        self.canvas = tk.Canvas(self, bg="white",
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 划分迷宫单元格，即根据坐标位置划线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 起点位置
        origin = np.array([20, 20])

        # 创建陷阱1
        trap1_center = origin + np.array([UNIT * 2, UNIT])
        self.trap1 = self.canvas.create_oval(
            trap1_center[0] - 15, trap1_center[1] - 15,
            trap1_center[0] + 15, trap1_center[1] + 15,
            fill='black')

        # 创建陷阱2
        trap2_center = origin + np.array([UNIT, UNIT * 2])
        self.trap2 = self.canvas.create_oval(
            trap2_center[0] - 15, trap2_center[1] - 15,
            trap2_center[0] + 15, trap2_center[1] + 15,
            fill='black')

        # 创建宝藏
        treasure_center = origin + UNIT * 2
        self.treasure = self.canvas.create_rectangle(
            treasure_center[0] - 15, treasure_center[1] - 15,
            treasure_center[0] + 15, treasure_center[1] + 15,
            fill='yellow')
        self.treasure_coord = self.canvas.coords(self.treasure)

        # 创建可以移动的红色格子代表勇士，并放置在起始位置
        self.warrior = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 组合所有元素
        self.canvas.pack()

    def reset(self):
        """重置迷宫界面
        """
        self.update()  # 更新tkinter的配置
        time.sleep(0.5)

        # 删除当前勇士的位置，重置其回到起点
        self.canvas.delete(self.warrior)
        origin = np.array([20, 20])
        self.warrior = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 返回勇士的起始位置
        init_coord = self.canvas.coords(self.warrior)
        print(init_coord)
        print(self.treasure_coord)
        dist_to_treasure = np.array(init_coord)[:self.n_features] - np.array(self.treasure_coord)[:self.n_features]
        init_state = dist_to_treasure / (MAZE_H * UNIT)
        return init_state

    def step(self, action):
        """根据动作，更新迷宫状态
        """
        state = self.canvas.coords(self.warrior)

        base_action = np.array([0, 0])
        if action == 0:     # 向上
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 向下
            if state[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 向左
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # 向右
            if state[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        # 移动勇士位置
        self.canvas.move(self.warrior, base_action[0], base_action[1])

        # 移动后勇士的位置
        next_coords = self.canvas.coords(self.warrior)

        # 奖励函数
        # 到达宝藏位置奖励1，到达陷阱处奖励-1，其他位置奖励0
        if next_coords == self.canvas.coords(self.treasure):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.trap1), self.canvas.coords(self.trap2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        
        # 计算到宝藏的距离
        dist_to_treasure = np.array(next_coords)[:self.n_features] - np.array(self.treasure_coord)[:self.n_features]
        state_next = dist_to_treasure / (MAZE_H * UNIT)

        return state_next, reward, done

    def render(self, time_interval=0.05):
        time.sleep(time_interval)
        self.update()


# **ReplayMemory**

# 记忆池：用于存储多条经历过的经验，能够实现记忆回放
# 
# 记忆池大小是固定的，如果超出池子大小，旧记忆会被新记忆替换

# In[5]:


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)  # 定义一个队列用于存放记忆，实现记忆池的自动伸缩

    def store_memory(self, state, action, reward, state_next, done):
        exp = [state, action, reward, state_next, 1 - int(done)]
        self.buffer.append(exp)

    def sample(self, batch_size):
        """采样记忆，并返回格式为ndarray的数据"""
        mini_batch = random.sample(self.buffer, batch_size)  # 从记忆池中随机挑选一定数量的记忆
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_)
            done_batch.append(done)

        obs_batch = np.array(obs_batch).astype("float32")
        action_batch = np.array(action_batch).astype("int32")
        reward_batch = np.array(reward_batch).astype("float32")
        next_obs_batch = np.array(next_obs_batch).astype("float32")
        done_batch = np.array(done_batch).astype("float32")

        return (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


# **搭建神经网络**

# 这里我们需要定义DQN所有的计算逻辑，其成员函数的功能如下：
# 1. learn：定义DQN如何学习和更新参数的，这里涉及了target网络和predict网络的交互使用
# 2. build_net：定义两个网络的结构，网络结构并不复杂，只是两层的Dense层的全连接网络
# 3. train_model：定义predict网络如何进行梯度更新
# 4. build_net：构建两个神经网络。我们需要与宝藏的距离越来越小，所以这里我们选择MSE损失来计算当前位置与宝藏的误差

# In[6]:


class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.9, replace_target_steps=200,
                 fc1_dims=32, fc2_dims=16):
        self.n_actions = n_actions  # 行为个数
        self.n_features = n_features  # 输入的特征个数
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减因子

        # 创建target网络和predict网络
        self.build_net(fc1_dims, fc2_dims)

        # 优化器和损失函数
        self.predict_model.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.predict_model.loss_func = tf.losses.MeanSquaredError()

        self.global_steps = 0  # 网络训练的总步数
        self.replace_target_steps = replace_target_steps  # 更换target网络的步数

    def build_net(self, fc1_dims=128, fc2_dims=64):
        # DQN的网络结构并不复杂，两层Dense层+输出层
        # 主网络，及时更新参数
        predict_model = keras.Sequential([
            keras.layers.Input(shape=(self.n_features)),
            keras.layers.Dense(fc1_dims, activation="relu", name="p1"),
            keras.layers.Dense(fc2_dims, activation="relu", name="p2"),
            keras.layers.Dense(self.n_actions, activation=None, name="p3")
        ], name="predict")
        predict_model.summary()
        self.predict_model = predict_model

        # target网络，定期从主网络更新参数
        target_model = keras.Sequential([
            keras.layers.Input(shape=(self.n_features)),
            keras.layers.Dense(fc1_dims, activation="relu", name="t1"),
            keras.layers.Dense(fc2_dims, activation="relu", name="t2"),
            keras.layers.Dense(self.n_actions, activation=None, name="t3")
        ], name="target")
        target_model.summary()
        self.target_model = target_model

    def replace_weights(self):
        """eval模型权重更新到target模型权重"""
        self.target_model.get_layer("t1").set_weights(self.predict_model.get_layer("p1").get_weights())
        self.target_model.get_layer("t2").set_weights(self.predict_model.get_layer("p2").get_weights())
        self.target_model.get_layer("t3").set_weights(self.predict_model.get_layer("p3").get_weights())

    def train_model(self, action, features, labels):
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.predict_model(features, training=True)
            # 根据我们需要的action，挑选出对应的Q值
            pred_action_value = tf.gather_nd(predictions, indices=list(enumerate(action)))
            loss = self.predict_model.loss_func(labels, pred_action_value)

        tvars = self.predict_model.trainable_variables
        gradients = tape.gradient(loss, tvars)
        self.predict_model.optimizer.apply_gradients(zip(gradients, tvars))

    def learn(self, batch_data):
        """使用DQN算法更新eval模型的权重
        """
        (state, action, reward, state_next, done) = batch_data

        # 每隔一定训练步数同步一次eval模型和target模型的参数
        if self.global_steps % self.replace_target_steps == 0:
            self.replace_weights()

        # 从target_model中获取max Q'的值，用于计算target_Q
        next_pred_value = self.target_model(state_next, training=False)
        best_value = tf.reduce_max(next_pred_value, axis=1)
        done = tf.cast(done, dtype=tf.float32)
        # 当到达terminal时，done值为1，target = R
        # 还未到达terminal时，done值为0，target = R + gamma * max(Q)
        target = reward + self.gamma * (1.0 - done) * best_value

        self.train_model(action, state, target)
        self.global_steps += 1


# **Agent**

# 智能体Agent负责DQN算法与环境的交互，在交互过程中把生成的数据提供给DQN来更新模型权重，数据的预处理流程也一般定义在这里。

# In[51]:


class Agent:
    def __init__(self, n_actions, network, epsilon_greedy=0.9, epsilon_greedy_increment=1e-6):
        self.epsilon_greedy_increment = epsilon_greedy_increment  # 贪婪度的增量
        self.epsilon = epsilon_greedy # 贪婪度
        self.network = network
        self.n_actions = n_actions

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            action = self.predict_best_action(observation)  # 选择最优动作
            # 随着训练逐步收敛，探索的程度慢慢降低，即贪婪程度逐渐增加
            self.epsilon = min(0.99, self.epsilon + self.epsilon_greedy_increment)
        else:
            action = np.random.randint(0, self.n_actions)  # 探索：每个动作都有概率被选择
        return action

    def predict_best_action(self, observation):
        """基于当前的观测值，评估网络给出最佳行为"""
        observation = tf.expand_dims(observation, axis=0)
        action = self.network.predict_model(observation, training=False)
        action = np.argmax(action)
        return action


# DQN于环境交互最重要的部分，大致流程与Q表格学习一致。这里我们定义单个回合的计算逻辑

# In[7]:


LEARN_FREQ = 5  # 训练频率，不需要每一个step都学习，攒一些新增经验后再学习，提高效率
MEMORY_SIZE = 2000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory里需要预存一些经验数据，再从里面抽样一个batch的经验让智能体去学习
BATCH_SIZE = 32  # 每次给智能体去学习的数据数量，从replay memory随机里抽样一批数据出来
LEARNING_RATE = 0.01  # 学习率
GAMMA = 0.9  # 奖励的衰减因子，一般取0.9到0.999不等
MAX_EPISODE = 300  # 游戏回合总数


# In[8]:


def run_episode(env, agent, memory, network):
    # 初始化环境
    observation = env.reset()
    steps = 0
    total_reward = 0

    while True:
        # 刷新环境
        # env.render()

        # 累计总步数
        steps += 1

        # DQN根据观测值选择行为
        action = agent.choose_action(observation)

        # 环境根据行为给出下一个状态和奖励，以及终止符
        observation_next, reward, done = env.step(action)

        # dqn存储记忆
        memory.store_memory(observation, action, reward, observation_next, done)

        # 控制学习的起始时间和频率
        # 先积累一些记忆后再开始学习，避免最开始训练的时候样本丰富度不够
        if (len(memory) > MEMORY_WARMUP_SIZE) and (steps % LEARN_FREQ == 0):
            network.learn(memory.sample(BATCH_SIZE))

        # 将下一个state变为下次循环的state
        observation = observation_next

        total_reward += reward

        # 如果触发终止，结束本回合
        if done:
            find_treasure = "Yes" if reward == 1 else "No"
            break

    return total_reward, steps, find_treasure


# DQN-Maze 主函数，增加一个评估环节，每N回合后评估下DQN的性能

# In[9]:


def evaluate(env, agent, render=False):
    eval_rewards = []

    for i in range(5):
        observation = env.reset()
        episode_reward = 0

        while True:
            action = agent.predict_best_action(observation)
            observation, reward, done = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if done:
                print(f"Eval episode {i} done")
                break
        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


# In[10]:


def main(env, agent, memory, dqn):
    # 开始训练
    num_fails = 0  # 记录失败的次数
    for i in range(MAX_EPISODE):  # 训练MAX_EPISODE个回合，其中eval部分不计入回合数
        # 训练
        total_reward, steps, find_treasure = run_episode(env, agent, memory, dqn)
        print(f"Episode {i}: use {steps} steps. Find treasure: {find_treasure}. Reward: {total_reward}")

        if find_treasure == "No":
            num_fails += 1

        # 每20个回合测试下DQN性能
        if i % 20 == 0 and i > 0:
            print("Evaluating...")
            eval_reward = evaluate(env, agent, render=False)
            print(f"Episode {i}: epsilon_greedy - {agent.epsilon}, eval_reward: {eval_reward}")

    # 打印失败次数
    print(f"Failed episode: {num_fails}/{MAX_EPISODE}")


# 与Q表一样的初始化环境和网络的步骤，只是将Q表换成了DQN
# 
# Tkinter暂不支持notebook显示，需要在命令行环境运行dqn_maze.py

# In[11]:


n_features = 2

# 初始化环境实例
env = Maze2(n_features)

# 实例化记忆池，DQN，智能体
memory = ReplayMemory(MEMORY_SIZE)
dqn = DeepQNetwork(env.n_actions, n_features)
agent = Agent(env.n_actions, dqn)

env.after(100, main(env, agent, memory, dqn))
env.mainloop()

print("Game over!")
env.destroy()


# ## 示例2——打砖块

# ### Breakout

# <p align=center><img src="./images/breakout1.png" alt="breakout1" style="width:400px;"/></p>

# 在这个游戏环境中，棋盘沿着屏幕底部移动，返回一个球，将摧毁屏幕顶部的块。
# 
# 游戏的目的是移除关卡中的所有障碍物并突破关卡。
# 
# 智能体必须学会通过左右移动和接球的方式，在球没有通过板的前提下移除所有块来控制板。

# Deepmind在论文训练了总共5000万帧（也就是总共大约38天的游戏体验）才最终达到了每次都能完成消除砖块的任务。
# 
# 然而，本实战的目标是在不到24小时内在一台具备显卡的机器上处理大约1000万帧时， 达到不错的结果

# ### Gym

# 2014年，Google DeepMind发表了一篇题为“Playing Atari with Deep Reinforcement Learning”的论文，可以在专家级的人类水平上玩 Atari 2600游戏。这是将深度神经网络应用于强化学习的第一个突破。

# [Gym](http://gym.openai.com/docs/) 由Open AI于2016年发布。它是一个用于开发和比较强化学习算法的工具包。

# Gym平台上有许多可用的游戏。并且您需要安装gym[atari]环境才能使Atari游戏可用

# ### 代码实现

# 首先我们引入一些环境依赖和tensorflow包

# In[37]:


from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# **环境**

# 全局的配置参数

# In[2]:


seed = 42  # 随机种子，确保能够复现出一样的结果
gamma = 0.99  # 奖励衰减因子
epsilon = 1.0  # 贪婪度
epsilon_min = 0.1  # 最小贪婪度
epsilon_max = 1.0  # 最大贪婪度
epsilon_interval = (epsilon_max - epsilon_min)  # 减少随机操作发生的几率
batch_size = 32  # 每次学习的数据大小
max_steps_per_episode = 10000  # 每个回合最多的游戏次数，即接球的次数


# 初始化环境
# 
# BreakoutNoFrameskip-v4：表示游戏为breakout；v4表示只执行智能体给出的行为，不会重复之前的行为；NoFrameskip表示没有跳帧

# gym的第一步通常是使用make方法得到一个游戏环境，如"Breakout-v4"。这个环境是抽象类env的实例化，通常具有以下属性和方法：
# 
# * observation_space：环境的状态空间；
# * action_space：环境的动作空间；
# * reset()：重置环境为初始状态，并且返回该初始状态；
# * step()：agent与环境交互的接口。它会接受一个动作，然后返回更新后的环境状态、奖励、结束标识符以及其它相关游戏信息

# 每一个状态都是一张游戏截图，分辨率为210 × 160 × 3，每一个像素取值在0~255

# In[23]:


# 由于 Deepmind 辅助函数使用OpenBaseline Atari 环境
env = make_atari("BreakoutNoFrameskip-v4")
print(env.observation_space)


# Wrapper是env的子类，是一个对env增加自定义功能的工具，这里我们使用Deepmind论文使用的wrapper

# In[24]:


# 增加灰度，放置四个帧并缩放到更小的比例
# 都是为了让输入变得更简单一些
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)


# 我们可以看到图像分辨率变为84 × 84 × 4，增加了灰度通道

# In[25]:


env.observation_space


# 我们可以看到有四种动作，分别为保持不动，攻击，向右移动，向左移动

# In[36]:


env.action_space


# In[41]:


env.unwrapped.get_action_meanings()


# **实现DQN网络**

# 该网络学习Q表的近似值，这是智能体将采取的状态和动作之间的映射。
# 
# 对于每个状态，我们将有四个可以采取的行动。
# 
# 游戏环境提供状态，并通过选择输出层中预测的四个Q值中较大的一个来选择动作。

# 网络结构：CNN图像分类
# 
# 网络的输入：当前帧数的游戏画面
# 
# 网络的输出：四种动作的概率值

# In[4]:


num_actions = env.action_space.n


def create_q_model():
    # Deepmind论文中定义的网络
    inputs = layers.Input(shape=(84, 84, 4,))

    # 屏幕上每一帧画面的卷积
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# 第一个模型对Q值进行预测，这些Q值用于帮助决策出一个动作
model = create_q_model()
# 建立一个target模型来预测未来的奖励。
# target模型的权重每10000步更新一次，因为这样计算出的Q值和目标Q值之间的损失是稳定的
model_target = create_q_model()


# 一些训练参数配置

# In[5]:


# 原文中使用的是RMSProp优化器，我们这里使用效果更好的Adam
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# 与Maze-DQN一样的记忆池，用于存放各种状态和行为及奖励
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []

running_reward = 0
episode_count = 0
frame_count = 0

# 前N帧数采取随机行动
epsilon_random_frames = 50000
# 贪婪度每一定帧数进行衰减
epsilon_greedy_frames = 1000000.0
# 记忆池容量，原文中建议1000000大小，但这样会有内存问题
max_memory_length = 100000
# 在采取4次行为后训练模型
update_after_actions = 4
# 多久更新一次target网络
update_target_network = 10000
# 使用Huber损失，来保证训练的稳定
loss_function = keras.losses.Huber()


# 基本上训练逻辑与之前的Maze一致，只是实现的代码方式上有一些出路
# 
# 如果需要展示图形界面，必须在命令行环境下运行，Notebook暂不支持可视化界面，需要注释掉env.render()

# In[43]:


while True:  # 一直运行直到问题被解决
    # 当前屏幕图像的像素值
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # 尝试在弹出窗口中的显示智能体的行为
        # env.render()
        frame_count += 1

        # 使用贪婪度来表示探索程度
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # 采取随机行动
            action = np.random.choice(num_actions)
        else:
            # 对环境中获取到的状态，预测下一个行为
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # 采用概率最大的行为
            action = tf.argmax(action_probs[0]).numpy()

        # 采取随机行动的衰减概率
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # 在我们的环境中应用采样出的行为，并获得下一个状态和奖励
        # state_next：采取行动后当前屏幕图像的像素值
        # reward：奖励，每当游戏得分增加时，该返回值会有一个正反馈
        # done：游戏是否结束
        # lives：额外信息，如当前回合玩家的剩余生命数
        state_next, reward, done, lives = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # 在记忆池中保存动作和状态
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # 每4帧更新一次网络，且每次使用batch_size个数的数据进行更新
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # 随机抽取一定数量的记忆信息，获取记忆池中的记忆下标
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # 使用列表推导式来获取记忆池中的记忆信息
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # 为采样出来的未来状态计算更新后的Q值
            # 为确保值的稳定，使用target网络进行计算
            future_rewards = model_target.predict(state_next_sample)
            # Q值 = 奖励 + 奖励衰减因子 * 期望的未来奖励
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # 如果是最后一帧，最后一个Q置为 -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # 创建一个遮盖张量，以便我们只计算更新后的Q值的损失
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # 基于旧状态和更新的Q值训练模型
                q_values = model(state_sample)

                # 将遮盖张量应用于Q值以获得对应采取行为的Q值
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # 计算新的Q值和旧的Q值之间的误差
                loss = loss_function(updated_q_values, q_action)

            # 反向传播
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # 使用新的权重更新target模型
            model_target.set_weights(model.get_weights())
            # 打印信息
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # 限制记忆池的大小
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # 更新奖励列表
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:  # 任务终止条件
        print("Solved at episode {}!".format(episode_count))
        break


# 到达1600000帧数后，平均奖励值来到10分左右，之后一直到8000000帧数，奖励值一直在10分徘徊

# ### 结果展示

# 训练前：
# ![Imgur](https://i.imgur.com/rRxXF4H.gif)
# 早期训练阶段：
# ![Imgur](https://i.imgur.com/X8ghdpL.gif)
# 后期训练解读：
# ![Imgur](https://i.imgur.com/Z1K6qBQ.gif)
