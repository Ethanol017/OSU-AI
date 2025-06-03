import gymnasium as gym
import torch
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(num_episodes,checkpoint_path):
    env = gym.make("osu_env/osu-v0",human_play_test=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor = models.Actor().to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    
    for episode in range(1,num_episodes):
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
            episode_reward += reward
            done = terminated or truncated
            if done:
                break
            state = next_state
        
        print(f"Episode {episode} Reward: {episode_reward}")
        
if __name__ == "__main__":
    test(num_episodes=10, checkpoint_path="model/episode_100.pth")