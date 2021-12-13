import math
import os
import datetime
import time
import pathlib
import random
import pickle
import torch
import numpy as np
import json
import copy
import pyflann

class ExplorationExploitationScheduler(object):
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""

    def __init__(self, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=1000000,
                 replay_memory_start_size=50000, max_frames=25000000):
        """
        Args:
            DQN: A DQN object
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                    self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames


    def get_eps(self, frame_number, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            frame_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        eps = 1
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2
        return eps

def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)
    return function




def beta_scheduler(beta_start, beta_frames):
    def function(frame_idx):
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    return function

def create_log_dir(args):
    log_dir = f"umem{args.use_mem}{args.mem_mode}"
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.c51:
        log_dir = log_dir + "c51-"
    if args.prioritized_replay:
        log_dir = log_dir + "per-"
    if args.dueling:
        log_dir = log_dir + "dueling-"
    if args.double:
        log_dir = log_dir + "double-"
    if args.noisy:
        log_dir = log_dir + "noisy-"
    log_dir = log_dir + "dqn-"
    
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now
    if args.skip==4:
        skip = ""
    else:
        skip = "-"+str(args.skip)
    save_dir = os.path.join("runs", args.env+skip)
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    log_dir = os.path.join(save_dir, log_dir+args.env)
    return log_dir

def print_log(frame, prev_frame, prev_time, reward_list, length_list, loss_list, write_interval=0):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_loss = np.mean(loss_list) if len(loss_list) != 0 else 0.

    print("Frame: {:<8} FPS: {:.2f} Avg. Reward: {:.2f} Avg. Length: {:.2f} Avg. Interval: {:.2f}  Avg. Loss: {:.2f}".format(
        frame, fps, avg_reward, avg_length, write_interval, avg_loss
    ))
    return avg_reward

def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

def save_model(model, args):
    fname = f"umem{args.use_mem}{args.mem_mode}"
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.c51:
        fname += "c51-"
    if args.prioritized_replay:
        fname += "per-"
    if args.dueling:
        fname += "dueling-"
    if args.double:
        fname += "double-"
    if args.noisy:
        fname += "noisy-"



    if os.path.isdir(args.save_model) is False:
        os.mkdir(args.save_model)
    if args.skip==4:
        skip = ""
    else:
        skip = "-"+ str(args.skip)
    save_dir = os.path.join("model", args.env+skip)
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, f'{args.use_mem}args.jon'), 'w') as fp:
        sa = copy.copy(args)
        sa.device=None
        sa.replay_buffer=None
        json.dump(vars(sa), fp)
    save_dir = os.path.join(save_dir, f'{fname}.pt')
    torch.save(model.state_dict(), save_dir)
    if model.use_mem==1:
        with open(save_dir+'mem', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(model.trjmem, output)

    return  save_dir

def load_model(model, args):
    if args.load_model is not None:
        fname = os.path.join("models", args.load_model)
    else:
        fname = f"umem{args.use_mem}{args.mem_mode}"
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.c51:
            fname += "c51-"
        if args.prioritized_replay:
            fname += "per-"
        if args.dueling:
            fname += "dueling-"
        if args.double:
            fname += "double-"
        if args.noisy:
            fname += "noisy-"

        if args.skip == 4:
            skip = ""
        else:
            skip = "-" +str(args.skip)
        save_dir = os.path.join("model", args.env  +  skip)
        save_dir = os.path.join(save_dir, f'{fname}.pt')
        fname = save_dir

    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    model.load_state_dict(torch.load(fname, map_location))
    if model.use_mem==1:
        with open(save_dir+'mem', 'rb') as output:  # Overwrites any existing file.
            model.trjmem = pickle.load(output)

            model.trjmem.kdtree =  pyflann.FLANN()
            model.trjmem.kdtree.build_index(model.trjmem.keys.data.cpu().numpy(), algorithm='kdtree')
            print('load mem')
            print(model.trjmem.get_mem_size())

def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
