import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
import trjmem
import controller

from functools import partial


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # weight_shape = list(m.weight.data.size())
        # fan_in = np.prod(weight_shape[1:4])
        # fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        # w_bound = np.sqrt(6. / (fan_in + fan_out))
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1 and not classname.find('Noisy')!=-1:
        # weight_shape = list(m.weight.data.size())
        # fan_in = weight_shape[1]
        # fan_out = weight_shape[0]
        # w_bound = np.sqrt(6. / (fan_in + fan_out))
        # m.weight.data.uniform_(-w_bound, w_bound)
        nn.init.kaiming_normal_(m.weight,  nonlinearity='relu')
        m.bias.data.fill_(0)

def DQN(env, args):
    if args.c51:
        if args.dueling:
            model = CategoricalDuelingDQN(env, args.noisy, args.sigma_init,
                                          args.Vmin, args.Vmax, args.num_atoms, args.batch_size, args)
        else:
            model = CategoricalDQN(env, args.noisy, args.sigma_init,
                                   args.Vmin, args.Vmax, args.num_atoms, args.batch_size, args)
    else:
        if args.dueling:
            model = DuelingDQN(env, args.noisy, args.sigma_init, args)
        else:
            model = DQNBase(env, args.noisy, args.sigma_init, args)
            
    return model

def inverse_distance(h, h_i, epsilon=1e-3):
    return 1 / (torch.dist(h, h_i) + epsilon)

mse_criterion = nn.SmoothL1Loss()

USE_CUDA = torch.cuda.is_available()


class DQNBase(nn.Module):
    """
    Basic DQN + NoisyNet

    Noisy Networks for Exploration
    https://arxiv.org/abs/1706.10295
    
    parameters
    ---------
    env         environment(openai gym)
    noisy       boolean value for NoisyNet. 
                If this is set to True, self.Linear will be NoisyLinear module
    """
    def __init__(self, env, noisy, sigma_init, args=None):
        super(DQNBase, self).__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_inputs = self.input_shape[0]
        self.num_actions = env.action_space.n
        self.noisy = noisy
        self.use_mem = False
        self.model_name = args.model_name
        self.num_warm_up = args.num_warm_up
        self.gamma = args.gamma
        self.last_inserts = []
        self.insert_size = args.insert_size
        self.bstr_rate = args.bstr_rate
        self.mem_mode = args.mem_mode
        self.m_contr = []


        if noisy:
            self.Linear = partial(NoisyLinear, sigma_init=sigma_init)
        else:
            self.Linear = nn.Linear

        self.flatten = Flatten()

        if len(self.input_shape)>1:
            print("use CNN features")
            self.features = nn.Sequential(
                nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 1024, kernel_size=3, stride=1),
                nn.ReLU(),
            )

        else:
            print("use MLP features")
            self.features = nn.Sequential(
                nn.Linear(self.input_shape[0], 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU())


        if args.use_mem==1:
            self.use_mem = True
            self.emb_index2count = {}
            self.replay_buffer = args.replay_buffer
            self.proj = nn.Linear(self._feature_size(), 32)

            self.beta = nn.Parameter(torch.tensor(0.0),
                                     requires_grad=True)
            self.fix_beta = args.fix_beta
            self.beta_net = nn.Linear(args.hidden_size, 1)
            self.feature_size = 32
            self.random_map = nn.Sequential(
                nn.Linear(args.hidden_size, args.mem_dim),
            )

            for param in self.random_map.parameters():
                param.requires_grad = False

            self.trjmem = trjmem.TrjMemory(inverse_distance, num_neighbors=args.k, max_memory=args.memory_size, lr=args.write_lr)
            self.trj_model = controller.LSTMController(self.feature_size + self.num_actions, args.hidden_size, num_layers=1)
            self.trj_out = nn.Linear(args.hidden_size, self.feature_size + self.num_actions +1)
            self.reward_model = nn.Sequential(
                nn.Linear(self.feature_size + self.num_actions + args.hidden_size, args.batch_size_reward),
                nn.ReLU(),
                nn.Linear(args.batch_size_reward, args.batch_size_reward),
                nn.ReLU(),
                nn.Linear(args.batch_size_reward, 1),
            )



        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions)
        )

        self.apply(weights_init)

    def clone_mem(self, target):
        target.trj_model.load_state_dict(self.trj_model.state_dict())
        target.trj_out.load_state_dict(self.trj_out.state_dict())
        target.trjmem = self.trjmem
        target.beta = self.beta
        target.emb_index2count = self.emb_index2count
        target.replay_buffer = self.replay_buffer

    def forward(self, x, h_trj=None, episode=0, use_mem=1.0, target=0, r=None, a=None, is_learning=False):

        if self.use_mem:
            return self.forward_mem(x, h_trj, episode=episode, use_mem=1.0, target=0,r=r, a=a, is_learning=is_learning)
        else:
            x = self.features(x)
            x = self.flatten(x)
            x = self.fc(x)
        return x
    
    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def forward_mem_feature(self, x, h_trj, episode=0, use_mem=1.0, target=0, r=None, a=None, is_learning=False):
        q_episodic = torch.zeros(x.shape[0], self.num_actions)
        if USE_CUDA:
            q_episodic = q_episodic.cuda()
        if episode > self.num_warm_up and random.random() < use_mem and self.trjmem.get_mem_size() > 1:

            if a is not None:
                _, h_trj_a = self.trj_model(self.make_trj_input(x, a), h_trj)
                q_episodic[:, a] = r+ self.gamma*self.episodic_net(h_trj_a, is_learning)
            else:
                for a in range(self.num_actions):
                    lxx, h_trj_a = self.trj_model(self.make_trj_input(x, a), h_trj)
                    if r is None:
                        r = self.reward_model(
                            torch.cat([self.make_trj_input(x, a), h_trj[0][0].detach()], dim=-1))

                    r = r.to(device=lxx.device).squeeze(-1)
                    q_episodic[:, a] = r + self.gamma*self.episodic_net(h_trj_a, is_learning)

                if is_learning is False and random.random()<self.bstr_rate:#REFINE
                    curV = q_episodic.max(1)[0]
                    if len(curV.shape)==1:
                        curV = curV.unsqueeze(-1)

                    self.trjmem.write(self.random_map(h_trj[0][0]), curV)


        q_episodic = q_episodic.detach()

       

        if self.fix_beta>0:
            b = self.fix_beta
        else:
            b = self.beta_net(h_trj[0][0])
            b = F.sigmoid(b)


        return q_episodic, b,  x

    def forward_mem(self, x, h_trj, episode=0, use_mem=1.0, target=0, r=None, a=None, is_learning=False):
        x = self.features(x)
        x = self.flatten(x)
        q_value_semantic = self.semantic_net(x, None, target)
        if self.use_mem:
            x = self.proj(x)

        q_episodic, b ,x =  self.forward_mem_feature(x, h_trj, episode, use_mem, target, r=r, a=a, is_learning=is_learning)
        return q_episodic * b + q_value_semantic, q_value_semantic, q_episodic, x



    def semantic_net(self, x, h_trj = None, q_episodic=0, target=0):
        if h_trj is not None:
            x = self.proj(x)
            _, h_trj = self.trj_model(self.make_trj_input(x, None), h_trj)
            q = self.fc(h_trj[0][0])
        else:
            q = self.fc(x)
        return q

    def episodic_net(self, h_trj, is_learning=False):
        fh_trj = self.random_map(h_trj[0][0])
        fh_trj = torch.as_tensor(fh_trj)

        q_estimates = self.trjmem.read(fh_trj, is_learning=is_learning)
        return q_estimates

    def make_trj_input(self, x, a):
        a_vec = torch.zeros(x.shape[0],self.num_actions)
        a_vec[:,a] = 1
        if USE_CUDA:
            a_vec = a_vec.cuda()

        x = torch.cat([x, a_vec],dim=-1)
        return x


    def add_trj(self, h_trj, R, step, episode):
        h_trj = h_trj[0][0]
        hkey = torch.as_tensor(h_trj)
        if USE_CUDA:
            hkey = hkey.cuda()
        hkey = self.random_map(hkey)
        embedding_index = self.trjmem.get_index(hkey)
        if embedding_index is None:
            self.trjmem.insert(hkey, R.unsqueeze(0))

            if self.insert_size>0:
                if len(self.last_inserts) > self.insert_size:
                    self.last_inserts.sort()
                    self.last_inserts = self.last_inserts[1:-1]
                self.last_inserts.append(R.unsqueeze(0))
            if episode>self.num_warm_up:
                try:
                    self.trjmem.write(hkey, R.unsqueeze(0))
                except Exception as e:
                    print(e)
        else:
            if episode>self.num_warm_up:
                try:
                    self.trjmem.write(hkey, R.unsqueeze(0))
                except Exception as e:
                    print(e)

    def compute_reward_loss(self, last_h_trj, traj_buffer, optimizer, batch_size, noise=0.1):
        sasr = random.choices(traj_buffer, k=batch_size-1)
        sasr.append(traj_buffer[-1])

        X = []
        y = []
        y2 = []
        hs1 = []
        hs2 = []
        for s1,h, a,s2,r in sasr:
            s1 = torch.as_tensor(s1).float()
            s2 = torch.as_tensor(s2).float()
            r = torch.FloatTensor([r]).unsqueeze(0)

            if USE_CUDA:
                s1 = s1.cuda()
                s2 = s2.cuda()
            if len(s1.shape) == 1:
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)

            x = self.features(s1.unsqueeze(0))
            x = self.flatten(x)
            x = self.proj(x)
            x = self.make_trj_input(x, a)

            x2 = self.features(s2.unsqueeze(0))
            x2 = self.flatten(x2)
            x2 = self.proj(x2)
            x2 = self.make_trj_input(x2, a)

            if noise>0:
                if random.random()>0.5:
                    X.append(F.dropout(x, p=noise))
                else:
                    noise_tensor = ((torch.max(torch.abs(x))*noise)**0.5)*torch.randn(x.shape).to(device=x.device)
                    if USE_CUDA:
                       noise_tensor = torch.tensor(noise_tensor).cuda()
                    X.append(x + noise_tensor.float())
            else:
                X.append(x)

            y.append(x)

            y2.append(torch.cat([x2, r.to(device=x2.device)], dim=-1))

            hs1.append(torch.tensor(h[0]).to(device=last_h_trj[0].device))
            hs2.append(torch.tensor(h[1]).to(device=last_h_trj[0].device))
        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        y2 = torch.stack(y2, dim=0).squeeze(1)
        cur_h_trj = (torch.cat(hs1, dim=1),
                     torch.cat(hs2, dim=1))


        pr = self.reward_model(torch.cat([X.squeeze(1), cur_h_trj[0][0]],dim=-1))
        l2 = mse_criterion(pr, y2[:, self.feature_size + self.num_actions:])
        optimizer.zero_grad()
        l2.backward()
        optimizer.step()
        return l2

    def compute_rec_loss(self, last_h_trj, traj_buffer, optimizer, batch_size, noise=0.5):
        sas = random.choices(traj_buffer, k=batch_size-1)
        sas.append(traj_buffer[-1])
        X = []
        y = []
        y2 = []
        hs1 = []
        hs2 = []
        for s1,h, a,s2, r in sas:
            s1 = torch.as_tensor(s1).float()
            s2 = torch.as_tensor(s2).float()
            r = torch.FloatTensor([r]).unsqueeze(0)

            if USE_CUDA:
                s1 = s1.cuda()
                s2 = s2.cuda()
            if len(s1.shape)==1:
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)

            x = self.features(s1.unsqueeze(0))
            x = self.flatten(x)
            x = self.proj(x)
            x = self.make_trj_input(x, a)

            x2 = self.features(s2.unsqueeze(0))
            x2 = self.flatten(x2)
            x2 = self.proj(x2)
            x2 = self.make_trj_input(x2, a)

            if random.random()>0.5:
                X.append(F.dropout(x, p=noise))
            else:
                noise_tensor = ((torch.max(torch.abs(x))*noise)**0.5)*torch.randn(x.shape).to(device=x.device)
                if USE_CUDA:
                    noise_tensor = torch.tensor(noise_tensor).cuda()
                X.append(x + noise_tensor.float())

            y.append(x)
            y2.append(torch.cat([x2, r.to(device=x2.device)], dim=-1))

            hs1.append(torch.tensor(h[0]).to(device=last_h_trj[0].device))
            hs2.append(torch.tensor(h[1]).to(device=last_h_trj[0].device))

        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        y2 = torch.stack(y2, dim=0).squeeze(1)

        last_h_trj = (last_h_trj[0].repeat(1, batch_size, 1), last_h_trj[1].repeat(1, batch_size, 1))

        y_p, _ = self.trj_model(X.squeeze(1), last_h_trj)
        y_p = self.trj_out(y_p)

        l1 = mse_criterion(y_p[:, :self.feature_size + self.num_actions], y2[:, :self.feature_size + self.num_actions])


        optimizer.zero_grad()
        l1.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.parameters(), 10)
        optimizer.step()
        return l1
    
    def act(self, state, epsilon, h_trj =None, episode=0):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        state = state.unsqueeze(0)

        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            action_e = None
            with torch.no_grad():
                if self.use_mem:
                    q_value, qs, qe, x = self.forward(state, h_trj, episode, use_mem=1)
                    action_s = qs.max(1)[1].item()
                    action_e = qe.max(1)[1].item()
                else:
                    q_value = self.forward(state)
                action  = q_value.max(1)[1].item()

                if action_e is not None:
                    if action == action_e:
                        self.m_contr.append(1)
                    else:
                        self.m_contr.append(0)

        else:
            x = self.features(state)
            x = self.flatten(x)
            if self.use_mem:
                x = self.proj(x)

            action = random.randrange(self.num_actions)


        if self.use_mem:

            y_trj, h_trj = self.trj_model(self.make_trj_input(x, action), h_trj)
            return action, h_trj, y_trj

        return action

    def get_pivot_lastinsert(self):
        if len(self.last_inserts)>0:
            return min(self.last_inserts)
        else:
            return -10000000

    def update_noisy_modules(self):
        if self.noisy:
            self.noisy_modules = [module for module in self.modules() if isinstance(module, NoisyLinear)]
    
    def sample_noise(self):
        for module in self.noisy_modules:
            module.sample_noise()

    def remove_noise(self):
        for module in self.noisy_modules:
            module.remove_noise()


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, noisy, sigma_init, args):
        super(DuelingDQN, self).__init__(env, noisy, sigma_init, args)
        
        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, 1)
        )

    def forward(self, x, h_trj=None, episode=0, use_mem=1.0, target=0,  r=None, a=None, is_learning=False):
        if self.use_mem:
            return self.forward_mem(x, h_trj, episode=0, use_mem=1.0, target=0, r=r, a=a, is_learning=is_learning)
        else:
            x = self.features(x)
            x = self.flatten(x)
            advantage = self.advantage(x)
            value = self.value(x)
            return value + advantage - advantage.mean(1, keepdim=True)

    def semantic_net(self, x, h_trj=None, q_episodic=0, target=0):
        # x = self.features(x)
        # x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)




class CategoricalDQN(DQNBase):
    """
    A Distributional Perspective on Reinforcement Learning
    https://arxiv.org/abs/1707.06887
    """

    def __init__(self, env, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDQN, self).__init__(env, noisy, sigma_init)
    
        support = torch.linspace(Vmin, Vmax, num_atoms)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
            .unsqueeze(1).expand(batch_size, num_atoms)

        self.register_buffer('support', support)
        self.register_buffer('offset', offset)
        self.num_atoms = num_atoms

        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions * self.num_atoms),
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_dist = self.forward(state)
                q_value = (q_dist * self.support).sum(2)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class CategoricalDuelingDQN(CategoricalDQN):

    def __init__(self, env, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDuelingDQN, self).__init__(env, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size)
        
        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, num_atoms)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)

        advantage = self.advantage(x).view(-1, self.num_actions, self.num_atoms)
        value = self.value(x).view(-1, 1, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features 
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.register_buffer('sample_weight_in', torch.FloatTensor(in_features))
        self.register_buffer('sample_weight_out', torch.FloatTensor(out_features))
        self.register_buffer('sample_bias_out', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.sample_noise()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.bias_sigma.size(0)))

    def sample_noise(self):
        self.sample_weight_in = self._scale_noise(self.sample_weight_in)
        self.sample_weight_out = self._scale_noise(self.sample_weight_out)
        self.sample_bias_out = self._scale_noise(self.sample_bias_out)

        self.weight_epsilon.copy_(self.sample_weight_out.ger(self.sample_weight_in))
        self.bias_epsilon.copy_(self.sample_bias_out)
    
    def _scale_noise(self, x):
        x = x.normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x
