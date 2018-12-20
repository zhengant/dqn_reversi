import tensorflow as tf
import numpy as np

from ReversiAgent import ReversiAgent
from ReplayBuffer import ReplayBuffer
from ReversiGame import ReversiGame


def compute_epsilon(episodes):
    return 0.5 - min(1e-3 * episodes, 0.5 - 0.001)


def dqn_reversi(num_episodes=1000000, memory_capacity=10000, model_memory_capacity=10, batch_size=32, C1=2, C2=10, save_freq=1000):
    agent = ReversiAgent()
    experience_buffer = ReplayBuffer(memory_capacity)
    model_buffer = ReplayBuffer(model_memory_capacity)
    model_buffer.add_memory(agent.clone())

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(episode)

        game = ReversiGame(model_buffer.get_random_memory())
        state = game.reset()
        done = False

        while not done:
            action = agent.get_next_move(state, compute_epsilon(episode))
            new_state, reward, done, _ = game.step(action)

            experience_buffer.add_memory((state, action, reward, new_state, done))

            if len(experience_buffer) > batch_size:
                agent.update_Q(experience_buffer.get_memory_batch(batch_size))

            if episode % C1 == 0:
                agent.update_targetQ()

            if episode % C2 == 0:
                model_buffer.add_memory(agent.clone())

            if episode % save_freq == 0:
                agent.save('reversi_agent.h5')

            state = new_state

    agent.save('reversi_agent.h5')


def main():
    dqn_reversi()


if __name__ == '__main__':
    main()