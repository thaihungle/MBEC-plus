import torch
import torch.optim as optim

import os
from common.utils import load_model
from model import DQN
import numpy as np
import json
from matplotlib.pylab import plt #load plot library


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


def test(env, args, current_model=None):
    if current_model is None:
        save_dir = os.path.join(args.save_model, args.env)
        try:
            with open(os.path.join(save_dir, f'{args.use_mem}args.jon')) as fp:
                sa = DictX(json.load(fp))
                sa.double = args.double
                sa.dueling = args.dueling
            current_model = DQN(env, sa).to(args.device)
        except:
            current_model = DQN(env, args).to(args.device)

        load_model(current_model, args)
    current_model.eval()
    num_params = 0
    for p in current_model.parameters():
        num_params += p.data.view(-1).size(0)
    print(f"no params {num_params}")
    rewards = []
    lens = []
    for n in range(args.num_test):

        # print(n)

        episode_reward = 0
        episode_length = 0

        state = env.reset()

        if args.use_mem == 1:
            br = current_model.bstr_rate
            #current_model.bstr_rate = 0
            h_trj = current_model.trj_model.create_new_state(1)

        step=0
        imgs = []
        while True:
            if args.render:
                if "mini" in args.env.lower():
                    img = env.render(view='top')

                    # plt.imshow(img)
                    # plt.show()
                    # img = env.render()
                    #
                    # plt.imshow(img)
                    # plt.show()

                    if args.save_render and step==0:
                        if "mini" in args.env.lower():
                            img = env.render(view='top')

                            f = plt.figure()
                            plt.axis('off')
                            plt.imshow(img)
                            save_dir = os.path.join("render", args.env)
                            if os.path.isdir(save_dir) is False:
                                os.mkdir(save_dir)
                            f.savefig(f"{save_dir}/topview.pdf")

                            img = env.render()
                            f = plt.figure()
                            plt.axis('off')
                            plt.imshow(img)
                            save_dir = os.path.join("render", args.env)
                            if os.path.isdir(save_dir) is False:
                                os.mkdir(save_dir)
                            f.savefig(f"{save_dir}/frontview.pdf")

                else:
                    env.render()
            if args.save_render:
                if "mini" in args.env.lower():
                    img = env.render(view='top')

                else:
                    img = env.render("rgb_array")

                if step%2==0:
                    imgs.append(img)


            if args.use_mem == 0:
                action = current_model.act(torch.FloatTensor(state).to(args.device), 0.)
            else:
                action, h_trj, y_trj = current_model.act(torch.FloatTensor(state).to(args.device), 0, h_trj,
                                                          episode=1000000)

            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_reward += reward
            episode_length += 1
            step+=1
            # print(episode_length, episode_reward)
            if done:
                if args.use_mem == 1:
                    current_model.bstr_rate = br
                break
        if args.save_render and episode_reward==3:
            pimg = np.asarray(imgs)
            img = np.min(pimg, axis=0)
            f = plt.figure()
            plt.axis('off')
            plt.imshow(img)
            save_dir = os.path.join("render", args.env)
            if os.path.isdir(save_dir) is False:
                os.mkdir(save_dir)
            f.savefig(f"{save_dir}/steps.pdf")



            raise False

        print(n, episode_reward)
        rewards.append(episode_reward)
        lens.append(episode_length)
    r = np.mean(rewards)
    l = np.mean(lens)
    print("True Test Result - Reward {} Length {}".format(r, l))

    return r


def test_online(env, args, current_model, numt=100):

    for n in range(numt):
        episode_reward = 0
        episode_length = 0

        state = env.reset()
        rewards = []
        lens = []
        if args.use_mem==1:
            h_trj = current_model.trj_model.create_new_state(1)
        while True:
            if args.render and n==numt-1:
                env.render()
            if args.use_mem==0:
                action = current_model.act(torch.FloatTensor(state).to(args.device), 0.)
            else:
                action, nh_trj, y_trj = current_model.act(torch.FloatTensor(state).to(args.device), 0, h_trj,
                                                          episode=1000000)

            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break
        rewards.append(episode_reward)
        lens.append(episode_length)
    r = np.mean(rewards)
    l = np.mean(lens)
    print("Test Result - Reward {} Length {}".format(r, l))
    return r, l