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


# 接受超参数
hyper_paras = hyper_parameters()
parse_args_maddpg = hyper_paras.parse_args_maddpg()
# 实例化环境和算法
env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=200, continuous_actions=True, render_mode="human")
maddpg = MADDPG(parse_args=parse_args_maddpg)

maddpg.load_checkpoint()

# 最大 epoch
num_epoch = 10
# 一句游戏最大 step
max_step = 50

# epoch 循环
for epoch in range(num_epoch):
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

        # time.sleep(0.1)


env.close()

