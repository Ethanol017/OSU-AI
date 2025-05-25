import gymnasium as gym
import numpy as np
import osu_env
import models
import torch
import torch.optim as optim
import torch.nn.functional as F

import collections, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self,buffer_limit=50000):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst, dtype=torch.float).to(device), \
                torch.tensor(r_lst, dtype=torch.float).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                torch.tensor(done_mask_lst, dtype=torch.float).to(device)
    
    def size(self):
        return len(self.buffer)

def train(config):
    # Init
    env = gym.make("osu_env/osu-v0",human_play_test=False)
    replay_buffer = ReplayBuffer(config["buffer_size"])
    actor = models.Actor(log_std_min=config["log_std_min"],log_std_max=config["log_std_max"]).to(device)
    
    critic1 = models.Critic().to(device)
    critic2 = models.Critic().to(device)
    
    critic1_target = models.Critic().to(device)
    critic2_target = models.Critic().to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    
    values_optimizer = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=config["critic_lr"])
    policy_optimizer = optim.Adam(list(actor.parameters()), lr=config["actor_lr"])
    
    if config["alpha_autotune"]:
        log_alpha_cont = torch.tensor(np.log(config["alpha_cont"]), requires_grad=True).to(device)
        log_alpha_disc = torch.tensor(np.log(config["alpha_disc"]), requires_grad=True).to(device)
        alpha_cont_optimizer = optim.Adam([log_alpha_cont], lr=config["alpha_autotune_lr"])
        alpha_disc_optimizer = optim.Adam([log_alpha_disc], lr=config["alpha_autotune_lr"])
        alpha_cont = log_alpha_cont.exp().detach()
        alpha_disc = log_alpha_disc.exp().detach()
        alpha_cont_target = torch.tensor(-2.0).to(device) # -2 is -(continous dim)
        alpha_disc_target = torch.tensor(-np.log(2)).to(device) # -log(2) is -log(discrete dim)
    else:
        alpha_cont = torch.Tensor(config["alpha_cont"]).to(device)
        alpha_disc = torch.Tensor(config["alpha_disc"]).to(device)
    
    
    
    for episode in range(1,config["num_episodes"]):
        state,_ = env.reset()
        done = False
        episode_reward = 0
        while True: # play loop
            state_tensor = torch.tensor(state, dtype=torch.float).squeeze(0).to(device)
            
            continuous_action, discrete_action, _ , _  = actor.get_action(state_tensor)
            action = {
                "move_action": continuous_action.numpy(),
                "click_action": discrete_action.numpy()
            }
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            action_rb = np.concatenate(continuous_action.numpy(), discrete_action.numpy())
            replay_buffer.put((state,action_rb,reward,next_state,done))
            state = next_state
            episode_reward += reward
            if done:
                break
        if replay_buffer.size() >= config["batch_size"]:
            for update in range(config["update_pre_episode"]):
                b_state, b_action, b_reward, b_state_next, b_dones = replay_buffer.sample(config["batch_size"])
                # update critic
                with torch.no_grad(): # target value (next state)
                    
                    _ , _ ,cont_log_pi_prob, disc_log_pi_prob = actor.get_action(state_tensor)
                    log_pi_prob = alpha_cont * cont_log_pi_prob + alpha_disc * disc_log_pi_prob
                    
                    q1_next, q2_next = critic1_target(b_state_next, continuous_action, discrete_action), critic2_target(b_state_next, continuous_action, discrete_action)
                    min_q_next = torch.min(q1_next, q2_next)
                    target_value = b_reward + (1 - b_dones) * config["gamma"] * (min_q_next - log_pi_prob)
                
                q1, q2 = critic1(b_state, b_action["move_action"], b_action["click_action"]), critic2(b_state, b_action["move_action"], b_action["click_action"])
                critic1_loss = F.mse_loss(q1, target_value)
                critic2_loss = F.mse_loss(q2, target_value)
                critic_loss = (critic1_loss + critic2_loss) / 2
                values_optimizer.zero_grad()
                critic_loss.backward()
                values_optimizer.step()

                if update % config["update_actor_freq"] == 0:
                    for _ in range(config["update_actor_freq"]): 
                        # update actor
                        _ , _ ,cont_log_pi_prob, disc_log_pi_prob = actor.get_action(state_tensor)
                        log_pi_prob = alpha_cont * cont_log_pi_prob + alpha_disc * disc_log_pi_prob
                        q1, q2 = critic1(b_state, b_action["move_action"], b_action["click_action"]), critic2(b_state, b_action["move_action"], b_action["click_action"])
                        min_q = torch.min(q1, q2)
                        actor_loss = (log_pi_prob - min_q).mean()
                        policy_optimizer.zero_grad()
                        actor_loss.backward()
                        policy_optimizer.step()
            
                        # update alpha
                        if config["alpha_autotune"]:
                            alpha_cont_loss = -(log_alpha_cont * (cont_log_pi_prob - alpha_cont_target).detach()).mean()
                            alpha_disc_loss = -(log_alpha_disc * (disc_log_pi_prob - alpha_disc_target).detach()).mean()
                            
                            alpha_cont_optimizer.zero_grad()
                            alpha_disc_optimizer.zero_grad()
                            alpha_cont_loss.backward()
                            alpha_disc_loss.backward()
                            alpha_cont_optimizer.step()
                            alpha_disc_optimizer.step()
                            
                            alpha_cont = log_alpha_cont.exp().detach()
                            alpha_disc = log_alpha_disc.exp().detach()
                
                if update % config["update_target_freq"] == 0:
                    for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                        target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)
                    for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                        target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)

if __name__ == "__main__":
    config = {
        "num_episodes": 100,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 0.1,
        "batch_size": 32,
        "buffer_size": 10000,
        "update_target_every": 10,
        "critic_lr": 0.001,
        "actor_lr": 0.001,
        "log_std_min": -5,
        "log_std_max": 0,
        "alpha_cont": 0.2,
        "alpha_disc": 0.05,
        "alpha_autotune": True,
        "alpha_autotune_lr": 1e-4,
        "update_pre_episode": 100,
        "update_actor_freq": 2,
        "update_target_freq": 1,
        "tau": 0.005,
    }
    train(config)