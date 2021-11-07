import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import test
import time, os
import numpy as np
from collections import deque
import random
from common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log, load_model, save_model, ExplorationExploitationScheduler
from model import DQN
from common.replay_buffer_mem import ReplayBuffer, PrioritizedReplayBuffer


def train(env, env_test, args, writer):
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)

    args.replay_buffer = replay_buffer

    current_model = DQN(env, args).to(args.device)
    target_model = DQN(env, args).to(args.device)

    current_model.clone_mem(target_model)

    if args.noisy:
        current_model.update_noisy_modules()
        target_model.update_noisy_modules()

    if args.load_model and os.path.isfile(args.load_model):
        load_model(current_model, args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    eplan = ExplorationExploitationScheduler(replay_memory_start_size=args.learning_start,
                                              max_frames=args.max_frames_eps)

    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)


    state_deque = deque(maxlen=args.multi_step)
    reward_deque = deque(maxlen=args.multi_step)
    action_deque = deque(maxlen=args.multi_step)

    optimizer = optim.Adam(current_model.parameters(), lr=args.lr, eps=1e-4)
    # optimizer = optim.RMSprop(current_model.parameters(), lr=0.000625, alpha=0.95, momentum=0.95, eps=0.01)
    optimizer_rec = optim.Adam(current_model.parameters(), lr=args.lr)
    optimizer_reward = optim.Adam(current_model.reward_model.parameters())



    reward_list, length_list, loss_list = [], [], []

    episode_reward = 0
    true_episode_reward = 0
    episode_length = 0
    episode_num = 0

    prev_time = time.time()
    prev_frame = 1
    best_avg_reward = cur_avg_reward = -100000000000000

    state = env.reset()
    traj_buffer = []
    state_buffer = []
    aux_loss = []
    rec_l1s = []
    rec_l2s = []

    h_trj = current_model.trj_model.create_new_state(1)
    save_model(current_model, args)

    for frame_idx in tqdm(range(1, args.max_frames + 1)):
        if args.render:
            env.render()

        if args.noisy:
            current_model.sample_noise()
            target_model.sample_noise()

        epsilon = eplan.get_eps(frame_idx)
        action, nh_trj, y_trj = current_model.act(torch.FloatTensor(state).to(args.device), epsilon,h_trj, episode=episode_num)

        next_state, reward, done, _ = env.step(action)
        true_reward = reward
        if args.clip_rewards==1:
            reward = np.sign(reward)
        state_deque.append(state)
        reward_deque.append(reward)
        action_deque.append(action)

        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            with torch.no_grad():
                args.replay_buffer.push(n_state, (h_trj[0].detach().cpu().numpy(), h_trj[1].detach().cpu().numpy()),
                                        n_action, n_reward, next_state,
                                        (nh_trj[0].detach().cpu().numpy(), nh_trj[1].detach().cpu().numpy()), np.float32(done))


            traj_buffer.append((n_state, (h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()), n_action, next_state, reward))
            h_trj = nh_trj

        state = next_state
        episode_reward += reward
        true_episode_reward += true_reward
        episode_length += 1

        if done:
            state_buffer.append(((h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()), episode_reward, episode_length))

            add_mem = 0

            cc = 0
            for h, R, trj_step in state_buffer:
                RR = episode_reward - R
                if RR > args.min_reward:
                    if len(current_model.last_inserts) >= args.insert_size > 0 and RR < current_model.get_pivot_lastinsert():  # and RR<max(model.last_inserts):
                        pass
                    else:
                        current_model.add_trj(h, torch.as_tensor(RR), trj_step, episode_num)
                        add_mem = 1


                cc += 1
            if add_mem == 1:
                current_model.trjmem.commit_insert()

            if frame_idx < args.rec_period:
                l2 = current_model.compute_reward_loss(h_trj, traj_buffer, optimizer_reward, args.batch_size_reward,
                                               noise=0)
                rec_l2s.append(l2.data.item())
                l1 = current_model.compute_rec_loss(h_trj, traj_buffer, optimizer_rec, args.batch_size_plan,
                                                      noise=args.rec_noise)
                aux_loss.append((l1+l2).data.item())
                rec_l1s.append(l1.data.item())


            h_trj = current_model.trj_model.create_new_state(1)
            state = env.reset()

            reward_list.append(true_episode_reward)
            length_list.append(episode_length)
            writer.add_scalar("data/episode_length", episode_length, frame_idx)
            writer.add_scalar("data/episode_reward", true_episode_reward, frame_idx)

            writer.add_scalar('Mem/num stored', current_model.trjmem.get_mem_size(), int(frame_idx))
            # writer.add_scalar('Mem/alpha', F.sigmoid(torch.norm(current_model.alpha).detach()).item(), int(frame_idx))
            writer.add_scalar('Mem/min last', current_model.get_pivot_lastinsert(), int(frame_idx))
            if current_model.m_contr:
                writer.add_scalar('Mem/contrib', np.mean(current_model.m_contr), int(frame_idx))

            if len(aux_loss)>0:
                writer.add_scalar('Loss/aux loss', np.mean(aux_loss), int(frame_idx))
                writer.add_scalar('Loss/l1 loss', np.mean(rec_l1s), int(frame_idx))
                writer.add_scalar('Loss/l2 loss', np.mean(rec_l2s), int(frame_idx))

            episode_reward, episode_length, true_episode_reward = 0, 0, 0
            episode_num += 1
            state_deque.clear()
            reward_deque.clear()
            action_deque.clear()
            aux_loss = []
            rec_l1s = []
            rec_l2s = []
            state_buffer = []
            traj_buffer = []
            current_model.m_contr = []
            target_model.m_contr = []
        else:
            if frame_idx % args.write_interval == 0 and len(traj_buffer):
                state_buffer.append(((h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()), episode_reward, episode_length))
                if args.rec_rate>0:
                    l2 = current_model.compute_reward_loss(h_trj, traj_buffer, optimizer_reward, args.batch_size_reward,
                                                   noise=0)
                    rec_l2s.append(l2.data.item())
                if random.random() < args.rec_rate and frame_idx<args.rec_period:
                    l1 = current_model.compute_rec_loss(h_trj, traj_buffer, optimizer_rec, args.batch_size_plan, noise=args.rec_noise)
                    aux_loss.append((l1+l2).data.item())
                    rec_l1s.append(l1.data.item())

                h_trj = (h_trj[0].detach(), h_trj[1].detach())

        if args.train_freq>0 and len(replay_buffer) > args.learning_start and frame_idx % args.train_freq == 0:
            beta = beta_by_frame(frame_idx)
            loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta, episode_num)
            loss_list.append(loss.item())
            writer.add_scalar("Loss/td_loss", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(current_model, target_model)

        if frame_idx % args.evaluation_interval == 0:
            cur_avg_reward_tr = print_log(frame_idx, prev_frame, prev_time, reward_list, length_list, loss_list)
            print(f"task {args.env} best test {best_avg_reward}")
            reward_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            # save_model(current_model, args)
            # test.test_online(env, args, current_model, numt=5)

        if frame_idx % (args.evaluation_interval*10) == 0:
            test_args = args
            test_args.num_test = 10
            testr = test.test(env_test, test_args, current_model)
            current_model.train()

            cur_avg_reward = testr
            writer.add_scalar("data/test_episode_reward", testr, frame_idx)
            writer.add_scalar("data/test_episode_reward2", testr, episode_num)

            if cur_avg_reward > best_avg_reward:
                sdir = save_model(current_model, args)
                best_avg_reward = cur_avg_reward
                print(f"save model to {sdir}")

        current_model.clone_mem(target_model)




def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None, episode=None):
    """
    Calculate loss and optimize for non-c51 algorithm
    """
    if args.prioritized_replay:
        state, h_trj, action, reward, next_state, nh_trj, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, h_trj, action, reward, next_state, nh_trj, done = replay_buffer.sample(args.batch_size)

        weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    if not args.c51:

        hx = torch.tensor(h_trj[:, 0, 0, 0]).to(device=state.device).unsqueeze(0)
        cx = torch.tensor(h_trj[:, 1, 0, 0]).to(device=state.device).unsqueeze(0)
        q_values, q1, q2, _ = current_model(state, (hx, cx), episode, use_mem=1, target=0, r=reward, a=action, is_learning=True)
        nhx = torch.tensor(nh_trj[:, 0, 0, 0]).to(device=state.device).unsqueeze(0)
        ncx = torch.tensor(nh_trj[:, 1, 0, 0]).to(device=state.device).unsqueeze(0)
        # next_q_values = self.forward(next_state, (nhx, ncx), episode, use_mem=1)
        target_next_q_values, qn1, qn2, _ = target_model(next_state, (nhx, ncx), episode, use_mem=1, target=1, is_learning=True)


        # q_values = current_model(state)
        # target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if args.double:
            next_q_values,_,_,_ = current_model(next_state, (nhx, ncx), episode, use_mem=1, target=0, is_learning=True)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        #loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')

        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-5
        loss = (loss * weights).mean()
    
    else:
        q_dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        q_dist = q_dist.gather(1, action).squeeze(1)
        q_dist.data.clamp_(0.01, 0.99)

        target_dist = projection_distribution(current_model, target_model, next_state, reward, done, 
                                              target_model.support, target_model.offset, args)

        loss = - (target_dist * q_dist.log()).sum(1)
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-6
        loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    torch.nn.utils.clip_grad_norm(current_model.parameters(), args.clip_grad)
    optimizer.step()

    return loss


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)

    target_next_q_dist = target_model(next_state)

    if args.double:
        next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1, target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)

    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret