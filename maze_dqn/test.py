import time
import torch
import numpy as np
from maze_env import MazeEnv
from dqn_agent import DQNAgent


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    env = MazeEnv(size=6)
    agent = DQNAgent(
        state_size=env.state_space, action_size=env.action_space, device=device)
    agent.load("dqn_maze.pth")
    agent.epsilon = 0.0

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    print("初始状态:")
    env.render()

    while not done and steps < 100:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        action_names = ["上", "下", "左", "右"]
        print(f"第 {steps} 步: 动作={action_names[action]}")
        env.render()
        time.sleep(0.3)

    if done:
        print(f"成功到达终点！总步数: {steps}, 总奖励: {total_reward:.2f}")
    else:
        print(f"未到达终点，总步数: {steps}, 总奖励: {total_reward:.2f}")


if __name__ == "__main__":
    test()
