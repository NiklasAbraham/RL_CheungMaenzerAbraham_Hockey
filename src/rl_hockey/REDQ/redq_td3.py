import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from rl_hockey.common import Agent
from rl_hockey.REDQ.models import Actor, Critic
import copy
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins
    

def soft_update_model1_with_model2(model1, model2, rou):
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

class REDQTD3(Agent):
    """
    Naive TD3: num_Q = 2, num_min = 2
    REDQ: num_Q > 2, num_min = 2
    MaxMin: num_mins = num_Qs
    for above three variants, set q_target_mode to 'min' (default)
    Ensemble Average: set q_target_mode to 'ave'
    REM: set q_target_mode to 'rem'
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        **user_config
    ):
        self.config = {
            "polyak": 0.995,
            "utd_ratio": 20, 
            "num_Q": 10, 
            "num_min": 2, 
            "q_target_mode": 'min',
            "policy_update_delay": 20,
            "learning_rate": 3e-4,
            "max_action": 1.0,
            "discount": 0.99,
            "tau": 0.005,
            "expl_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "batch_size": 256,
            "latent_dim": [256, 256],
            "activation": nn.ReLU,
            "priority_replay": False,
            "normalize_obs": False
        }
        self.config.update(user_config)
        super().__init__(priority_replay=self.config["priority_replay"], normalize_obs=self.config["normalize_obs"])
        # set up networks
        self.actor = Actor(state_dim, action_dim, self.config["latent_dim"], self.config["activation"], self.config["max_action"]).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic_net_list, self.critic_target_net_list = [], []
        for q_i in range(self.config["num_Q"]):
            new_critic = Critic(state_dim, action_dim, self.config["latent_dim"], self.config["activation"]).to(DEVICE)
            self.critic_net_list.append(new_critic)
            new_critic_target = copy.deepcopy(new_critic)
            self.critic_target_net_list.append(new_critic_target)

        # set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["learning_rate"])
        self.critic_optimizer_list = []
        for q_i in range(self.config["num_Q"]):
            self.critic_optimizer_list.append(optim.Adam(self.critic_net_list[q_i].parameters(), lr=self.config["learning_rate"]))
        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = self.config["max_action"]
        self.lr = self.config["learning_rate"]
        self.hidden_sizes = self.config["latent_dim"]
        self.polyak = self.config["polyak"]
        self.batch_size = self.config["batch_size"]
        self.num_min = self.config["num_min"]
        self.num_Q = self.config["num_Q"]
        self.utd_ratio = self.config["utd_ratio"]
        self.q_target_mode = self.config["q_target_mode"]

        # TD3 arguments
        self.policy_update_delay = self.config["policy_update_delay"]
        self.max_action = self.config["max_action"]
        self.discount = self.config["discount"]
        self.tau = self.config["tau"]
        self.expl_noise = self.config["expl_noise"]
        self.policy_noise = self.config["policy_noise"]
        self.noise_clip = self.config["noise_clip"]
        self.policy_freq = self.config["policy_freq"]
        self.batch_size = self.config["batch_size"]
        self.priority_replay = self.config["priority_replay"]

    def __get_current_num_data(self):
        # used to determine whether we should get action from policy or take random starting actions
        return self.buffer.size

    def act(self, state, deterministic=False, t0s=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = self.actor(state).squeeze(0).cpu().numpy()
            if not deterministic:
                action += np.random.normal(
                    0, self.max_action * self.expl_noise, size=self.action_dim
                ).clip(-self.max_action, self.max_action)

            return action
        
    def act_batch(self, states, deterministic=False, t0s=None):
        """Process a batch of states at once (for vectorized environments)"""
        with torch.no_grad():
            states = torch.from_numpy(states).to(DEVICE)
            batch_size = states.shape[0]
            action = self.actor(states).squeeze(0).cpu().numpy()

            if not deterministic:
                action += np.random.normal(
                    0, self.max_action * self.expl_noise, size=(batch_size, self.action_dim)
                ).clip(-self.max_action, self.max_action)

            return action
        
    def evaluate(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            # 1. Get the deterministic action (No sampling, no log_prob)
            action = self.actor(state)

            # 2. Get the Q-values from both critics
            q_values = []
            for critic in self.critic_net_list:
                q_value = critic(state, action)
                q_values.append(q_value)

            # 3. Take the conservative estimate (min)
            value = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
            return value.squeeze().item()

    def get_action_and_logprob_for_bias_evaluation(self, state):
        # I think this method for TD3 is just using the act method
        return self.act(state, deterministic=False)

    def get_ave_q_prediction_for_bias_evaluation(self, state, action):
        # given state and act_tensor, output Q prediction
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.critic_net_list[q_i](state, action)
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def get_redq_critic_target_no_grad(self, next_state, reward, done):
        # compute REDQ Q target, depending on the agent's Q target mode
        # allow min as a float:
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            if self.q_target_mode == 'min':
                """Q target is min of a subset of Q values"""
                next_action = self.actor_target(next_state)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.critic_target_net_list[sample_idx](next_state, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                target_Q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
                
            if self.q_target_mode == 'ave':
                raise NotImplementedError("Ensemble Average Q target not implemented for TD3 and Hockey Env")
                """Q target is average of all Q values"""
                next_action = self.actor_target(next_state)
                q_prediction_next_list = []
                for q_i in range(self.num_Q):
                    q_prediction_next = self.critic_target_net_list[q_i](next_state, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_ave = torch.cat(q_prediction_next_list, 1).mean(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_prediction_next_ave - self.alpha * log_prob_next_action
                y_q = reward + self.gamma * (1 - done) * next_q_with_log_prob
            if self.q_target_mode == 'rem':
                raise NotImplementedError("REM Q target not implemented for TD3 and Hockey Env")
                """Q target is random ensemble mixture of Q values"""
                next_action = self.actor_target(next_state)
                q_prediction_next_list = []
                for q_i in range(self.num_Q):
                    q_prediction_next = self.critic_target_net_list[q_i](next_state, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                # apply rem here
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                rem_weight = Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.device)
                normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.num_Q)
                rem_weight = rem_weight / normalize_sum
                q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_prediction_next_rem - self.alpha * log_prob_next_action
        target_Q = reward + (1-done) * self.discount * target_Q
        return target_Q, sample_idxs

    def train(self, steps=20):
        critic_losses = []
        actor_losses = []

        for i_update in range(steps):
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

            state = torch.from_numpy(state).to(dtype=torch.float32, device=DEVICE)
            action = torch.from_numpy(action).to(dtype=torch.float32, device=DEVICE)
            reward = torch.from_numpy(reward).to(dtype=torch.float32, device=DEVICE)
            next_state = torch.from_numpy(next_state).to(
                dtype=torch.float32, device=DEVICE
            )
            done = torch.from_numpy(done).to(dtype=torch.float32, device=DEVICE)

            """Q loss"""
            target_Q, sample_idxs = self.get_redq_critic_target_no_grad(next_state, reward, done)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.critic_net_list[q_i](state, action)
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            target_Q = target_Q.expand((-1, self.num_Q)) if target_Q.shape[1] == 1 else target_Q
            q_loss_all = self.mse_criterion(q_prediction_cat, target_Q) * self.num_Q
            critic_losses.append(q_loss_all.item())

            for q_i in range(self.num_Q):
                self.critic_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """actor loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == steps - 1:
                # get actor loss
                next_action = self.actor(state)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.critic_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.critic_net_list[sample_idx](state, next_action)
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                actor_loss = -(ave_q).mean() # TD3 must negative sign
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_losses.append(actor_loss.item())
                for sample_idx in range(self.num_Q):
                    self.critic_net_list[sample_idx].requires_grad_(True)

            """update networks"""
            for q_i in range(self.num_Q):
                self.critic_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == steps - 1:
                self.actor_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.critic_target_net_list[q_i], self.critic_net_list[q_i], self.polyak)

        return {"critic_loss": critic_losses, "actor_loss": actor_losses}

    def save(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        if not filepath.endswith(".pt"):
            filepath += ".pt"

        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": [critic.state_dict() for critic in self.critic_net_list],
            "critic_target": [critic_target.state_dict() for critic_target in self.critic_target_net_list],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": [critic_optimizer.state_dict() for critic_optimizer in self.critic_optimizer_list],
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        for critic, state_dict in zip(self.critic_net_list, checkpoint["critic"]):
            critic.load_state_dict(state_dict)
        for critic_target, state_dict in zip(self.critic_target_net_list, checkpoint["critic_target"]):
            critic_target.load_state_dict(state_dict)
        for critic_optimizer, state_dict in zip(self.critic_optimizer_list, checkpoint["critic_optimizer"]):
            critic_optimizer.load_state_dict(state_dict)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
