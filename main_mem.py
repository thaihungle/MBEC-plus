import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time, os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds, read_config
from common.wrappers import make_atari, wrap_atari_dqn, make_3dnav
from arguments import get_args
from train_mem import train
from test import test
import json

# import pyvirtualdisplay
# # Creates a virtual display for OpenAI gym
# pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
# import gym_miniworld


class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


def main():
    args = get_args()
    args.num_test = args.evaluation_interval
    args.replay_buffer=None


    # args.clip_rewards=0
    try:
        save_dir = os.path.join(args.save_model, args.env)
        with open(os.path.join(save_dir, f'{args.use_mem}args.jon')) as fp:
            sa = DictX(json.load(fp))
            args.scale = sa.scale
    except:
        print("no json file use default param")
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)

    setup_json = read_config("config.json")
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    args.env_conf = env_conf
    if "world" in args.env.lower():
        env = make_3dnav(args.env, args)
        env_test = make_3dnav(args.env, args)
    elif "-v" in args.env:
        print("ATARI ENV")
        env = make_atari(args.env, args)
        env, env_test = wrap_atari_dqn(env, args)
    else:
        print("GYM ENV")
        env = gym.make(args.env)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        r = test(env_test, args)
        env.close()
        with open("./temp.txt", "w") as f:
            f.write(str(r))
        return r

    train(env, env_test, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
