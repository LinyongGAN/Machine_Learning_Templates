import random
import numpy as np
import torch
from tqdm import tqdm
from maze_env import MazeEnv
from dqn_agent import DQNAgent


def train():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    env = MazeEnv(size=6)
    agent = DQNAgent(
        state_size=env.state_space,
        action_size=env.action_space,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        device=device,
    )

    num_episodes = 500
    max_steps = 100
    batch_size = 64
    target_update_freq = 10

    rewards_history = []
    success_count = 0

    print("开始训练...")
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.learn(batch_size)

            state = next_state
            total_reward += reward

            if done:
                success_count += 1
                break

        agent.decay_epsilon()

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            success_rate = success_count / 50
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"成功率: {success_rate:.2f}, "
                  f"epsilon: {agent.epsilon:.4f}")
            success_count = 0

    agent.save("dqn_maze.pth")
    print("训练完成，模型已保存到 dqn_maze.pth")


if __name__ == "__main__":
    train()
