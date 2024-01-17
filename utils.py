import math
import torch
import random
import numpy as np

# 创建复用 functions
class functions():

    # 对 obs 数据进行处理
    @staticmethod
    def obs_dict_to_array(obs_dict):
        # 设置的 agent 数量为 2
        agent_0_state = obs_dict["agent_0"]
        agent_1_state = obs_dict["agent_1"]

        actor_state = np.vstack((agent_0_state, agent_1_state), dtype=np.float32)
        critic_state = np.concatenate((agent_0_state, agent_1_state), axis=0, dtype=np.float32)

        return critic_state, actor_state
    

    # 对 actons 处理成适合 MPE 的动作数据
    @staticmethod
    def action_array_to_dict(actions):
        action_env = {}
        action_env["agent_0"] = actions[0]
        action_env["agent_1"] = actions[1]

        return action_env
        




