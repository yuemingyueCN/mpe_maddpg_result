import argparse
"""
Notice:
    自行补充参数数据
"""
class hyper_parameters():

    def parse_args_maddpg(self):
        parser = argparse.ArgumentParser("MADDGP Framworks Hyper Parasmeters")
        parser.add_argument("--alpha", type=float, default=0.001, help="ActorNetwork learning rate")
        parser.add_argument("--beta", type=float, default=0.001, help="CriticNetwork learning rate")
        parser.add_argument("--actor_states_dims", type=list, default=[12,12], help="所有agents的输入ActorNetwork的维度 eg:[3, 3] 其中有2个agent的获取动作的状态信息维度分别为 3 3")
        parser.add_argument("--actor_fc1", type=int, default=64, help="ActorNetwork linear 1 output dims")
        parser.add_argument("--actor_fc2", type=int, default=32, help="ActorNetwork linear 2 output dims")
        parser.add_argument("--actor_fc3", type=int, default=32, help="ActorNetwork linear 3 output dims")
        parser.add_argument("--critic_fc1", type=int, default=64, help="CriticNetwork linear 1 output dims")
        parser.add_argument("--critic_fc2", type=int, default=32, help="CriticNetwork linear 2 output dims")
        parser.add_argument("--critic_fc3", type=int, default=32, help="CriticNetwork linear 3 output dims")
        parser.add_argument("--n_actions", type=int, default=[5,5], help="所有agents的动作空间维度 eg:[2,3] 有两个agent动作空间为 2 3")
        parser.add_argument("--n_agents",type=int, default=2, help="number of agents")
        parser.add_argument("--chkpt_dir", type=str, default='model/maddpg/', help="model save/load chkpt_dir eg':model/maddpg/'")
        parser.add_argument("--gamma", type=float, default=0.95, help="attenuation factor gamma gamma 需要仔细考虑 因为需要不同的 gamma 影响不同")
        parser.add_argument("--tau", type=float, default=0.01, help="soft update parameters")
        parser.add_argument("--buffer_max_size", type=int, default=100000, help="经验池最大数据容量")
        parser.add_argument("--buffer_batch_size", type=int , default=1024, help="maddpg learn batch_size")

        return parser.parse_args()
    
