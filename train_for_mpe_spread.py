import time
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# 引入算法和环境
from pettingzoo.mpe import simple_spread_v3
from maddpg.maddpg import MADDPG
# 引入工具文件
from utils import functions
from hyper_parameters import hyper_parameters

# 以当前时间为名创建一个子目录 用来区分不同的日志文件
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
# 创建 writer
writer = SummaryWriter(log_dir=f'writer/{current_time}')

# 接受超参数
hyper_paras = hyper_parameters()
parse_args_maddpg = hyper_paras.parse_args_maddpg()
# 实例化环境和算法
env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=200, continuous_actions=True, render_mode=None)
maddpg = MADDPG(parse_args=parse_args_maddpg)


# 最大 epoch
num_epoch = 50000
# 一句游戏最大 step
max_step = 50
# 用于记录每一次参数迭代的 loss
step_for_loss = 0
# 用于 reward 记录数据
reward_record_agent_0 = 0
reward_record_agent_1 = 0
max_reward = -1000000

"""
探索部分
填充一半的经验池
"""

print("exploration")

# 这里增加一个逻辑逻辑用来探索
# epoch 循环
for epoch in range(1000):
    print("epoch :", epoch)
    # 重置环境
    obs_env, infos = env.reset()
    # step 循环
    for step in range(max_step):

        # 转换状态空间的数据格式 dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        # 获取动作
        actions = maddpg.choose_action(actor_state)
        # 转换动作为 dict
        action_env = functions.action_array_to_dict(actions)
        # step 动作
        obs_next_env, rewards_env, terminations_env, truncations, infos = env.step(action_env)
        # 获取新的状态数据
        critic_state_next, actor_state_next = functions.obs_dict_to_array(obs_next_env)
        # 状态转移
        obs_env = obs_next_env
        rewards = [rewards_env["agent_0"], rewards_env["agent_1"]]
        terminal = [terminations_env["agent_0"], terminations_env["agent_1"]]
        # 存储数据
        maddpg.buffer.store_transition(
            critic_state, actor_state, actions, rewards, critic_state_next, actor_state_next, terminal
        )

print("begin training")

"""
训练部分
"""
# epoch 循环
for epoch in range(num_epoch):
    # 重置环境
    obs_env, infos = env.reset()
    # step 循环
    for step in range(max_step):

        step_for_loss = step_for_loss + 1

        # 转换状态空间的数据格式 dict array
        critic_state, actor_state = functions.obs_dict_to_array(obs_env)
        # 获取动作
        actions = maddpg.choose_action(actor_state)
        # 转换动作为 dict
        action_env = functions.action_array_to_dict(actions)
        # step 动作
        obs_next_env, rewards_env, terminations_env, truncations, infos = env.step(action_env)
        # 获取新的状态数据
        critic_state_next, actor_state_next = functions.obs_dict_to_array(obs_next_env)
        # 状态转移
        obs_env = obs_next_env
        rewards = [rewards_env["agent_0"], rewards_env["agent_1"]]
        terminal = [terminations_env["agent_0"], terminations_env["agent_1"]]
        # 存储数据
        maddpg.buffer.store_transition(
            critic_state, actor_state, actions, rewards, critic_state_next, actor_state_next, terminal
        )
        # 训练
        maddpg.learn(writer, step_for_loss)

        # 记录 reward 数据
        reward_record_agent_0 = reward_record_agent_0 + rewards[0]
        reward_record_agent_1 = reward_record_agent_1 + rewards[1]

    reward_record = reward_record_agent_0 + reward_record_agent_1

    print('Ep: {} sum_eward: {}'.format(epoch+1, reward_record))

    writer.add_scalar('reward/sum_reward', reward_record, (epoch+1))

    # 如果奖励之和大于之前的最大奖励之和，则保存模型
    if reward_record >= max_reward:
        max_reward = reward_record
        # 保存模型
        maddpg.save_checkpoint()

    # 记录最大累计奖励
    writer.add_scalar('reward/max_reward', max_reward, (epoch+1))

    # 一局游戏累计奖励清零
    reward_record_agent_0 = 0
    reward_record_agent_1 = 0


env.close()

