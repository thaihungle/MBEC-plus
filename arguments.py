import argparse
import torch
import random
import time

def get_args():
    parser = argparse.ArgumentParser(description='MBEC++(MBEC+DQN)')

    # Basic Arguments
    parser.add_argument('--seed', type=int, default=int(time.time())%1000,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Training Arguments
    parser.add_argument('--max-frames', type=int, default=10000000, metavar='STEPS',
                        help='Number of frames to train')
    parser.add_argument('--max-frames-eps', type=int, default=40000000, metavar='STEPS',
                        help='Number of frames to train')
    parser.add_argument('--max_episode_length', type=int, default=18000, metavar='STEPS',
                       help='Number of frames per episode')
    parser.add_argument('--buffer-size', type=int, default=1000000, metavar='CAPACITY',
                        help='Maximum memory buffer size')
    parser.add_argument('--update-target', type=int, default=1000, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=4, metavar='STEPS',
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--learning-start', type=int, default=50000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps_final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('--eps_decay', type=int, default=30000,
                        help='Adjustment parameter for epsilon')
    parser.add_argument('--use_mem', type=int, default=1,
                        help='use memory')
    parser.add_argument("--batch_size_plan", type=int, default=4,
                        help="batch size planning model")
    parser.add_argument("--batch_size_reward", type=int, default=4,
                        help="batch size planning model")
    parser.add_argument("--reward_hidden_size", type=int, default=8,
                        help="batch size planning model")
    parser.add_argument("--hidden_size", type=int, default=16,
                        help="RNN hidden")
    parser.add_argument("--mem_mode", type=str, default="",
                        help="memory type")
    parser.add_argument("--mem_dim", type=int, default=8,
                        help="memory size")
    parser.add_argument("--memory_size", type=int, default=10000,
                        help="memory size")
    parser.add_argument("--insert_size", type=int, default=-1,
                        help="insert size")
    parser.add_argument("--k", type=int, default=7,
                        help="num neighbor")
    parser.add_argument("--fix_beta", type=float, default=-1,
                        help="no dynamic consolidation")
    parser.add_argument("--min_reward", type=float, default=-1000000000,
                        help="minimum reward of env")
    parser.add_argument("--write_interval", type=int, default=200,
                        help="interval for memory writing")
    parser.add_argument("--write_interval_rate", type=float, default=0.1,
                        help="interval for memory writing")
    parser.add_argument("--read_interval_rate", type=float, default=0.2,
                        help="interval for memory reading")
    parser.add_argument("--write_lr", type=float, default=0.5,
                        help="learning rate of writing")
    parser.add_argument("--bstr_rate", type=float, default=0.1,
                        help="learning rate of writing")
    parser.add_argument("--rec_rate", type=float, default=0.1,
                        help="rate of reconstruction learning")
    parser.add_argument("--rec_noise", type=float, default=0.5,
                        help="rate of reconstruction noise")
    parser.add_argument("--rec_period", type=int, default=5e6,
                        help="period of reconstruction learning")
    parser.add_argument("--num_warm_up", type=int, default=20,
                        help="number of episode warming up memory")
    parser.add_argument("--run_id", default="r1",
                        help="r1,r2,r3")
    parser.add_argument("--model_name", default="MBECplus",
                        help="name")

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='enable prioritized experience replay')
    parser.add_argument('--c51', action='store_true',
                        help='enable categorical dqn')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')
    parser.add_argument('--Vmin', type=int, default=-10,
                        help='Minimum value of support for c51')
    parser.add_argument('--Vmax', type=int, default=10,
                        help='Maximum value of support for c51')
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='Number of atom for c51')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta-frames', type=int, default=100000,
                        help='End frame of beta schedule for prioritized replay')
    parser.add_argument('--sigma-init', type=float, default=0.4,
                        help='Sigma initialization value for NoisyNet')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='Environment Name')
    parser.add_argument('--episode-life', type=int, default=1,
                        help='Whether env has episode life(1) or not(0)')
    parser.add_argument('--clip-rewards', type=int, default=1,
                        help='Whether env clip rewards(1) or not(0)')
    parser.add_argument('--skip', type=int, default=4,
                        help='frame skip rate')
    parser.add_argument('--frame-stack', type=int, default=1,
                        help='Whether env stacks frame(1) or not(0)')
    parser.add_argument('--scale', type=int, default=1,
                        help='Whether env scales(1) or not(0)')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default=None,
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='model',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--save_render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--evaluation_interval', type=int, default=10000,
                        help='Frames for evaluation interval')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--clip_grad', type=int, default=100,
                        help='clip gradient')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args
