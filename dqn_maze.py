"""
Refer from
1. https://blog.csdn.net/weixin_44018458/article/details/108038507
2. https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5_Deep_Q_Network/maze_env.py
"""

import collections
import random
import time
import tkinter as tk
import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras

LEARN_FREQ = 5  # 训练频率，不需要每一个step都学习，攒一些新增经验后再学习，提高效率
MEMORY_SIZE = 2000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面抽样一个batch的经验让智能体去学习
BATCH_SIZE = 32  # 每次给智能体去学习的数据数量，从replay memory随机里抽样一批数据出来
LEARNING_RATE = 0.01  # 学习率
GAMMA = 0.9  # 奖励的衰减因子，一般取0.9到0.999不等
MAX_EPISODE = 300  # 游戏回合总数

UNIT = 40   # 每个迷宫单元格的宽度
MAZE_H = 4  # 迷宫的高度
MAZE_W = 4  # 迷宫的宽度


class Maze(tk.Tk, object):
    def __init__(self, n_features):
        super(Maze, self).__init__()
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

        dist_to_treasure = np.array(next_coords)[:self.n_features] - np.array(self.treasure_coord)[:self.n_features]
        state_next = dist_to_treasure / (MAZE_H * UNIT)

        return state_next, reward, done

    def render(self):
        # time.sleep(0.1)
        self.update()


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

        try:
            obs_batch = np.array(obs_batch).astype("float32")
            action_batch = np.array(action_batch).astype("int32")
            reward_batch = np.array(reward_batch).astype("float32")
            next_obs_batch = np.array(next_obs_batch).astype("float32")
            done_batch = np.array(done_batch).astype("float32")
        except:
            print(next_obs_batch)
            traceback.print_exc()

        return (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


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


def run_episode(env, agent, memory, network):
    # 初始化环境
    observation = env.reset()
    steps = 0
    total_reward = 0

    while True:
        # 刷新环境
        env.render()

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


if __name__ == "__main__":
    # 与Q表直接返回是否为terminal不同，DQN只能接受数字输入，我们这里重新定义特征
    # 由于，我们将勇士放置在左上角，勇士只能向下或向右走才能到达宝藏位置，于是我们将当前勇士的位置到宝藏距离作为新的特征
    n_features = 2

    # 初始化环境实例
    env = Maze(n_features)

    memory = ReplayMemory(MEMORY_SIZE)
    dqn = DeepQNetwork(env.n_actions, n_features)
    agent = Agent(env.n_actions, dqn)

    env.after(100, main(env, agent, memory, dqn))
    env.mainloop()

    print("Game over!")
    env.destroy()
