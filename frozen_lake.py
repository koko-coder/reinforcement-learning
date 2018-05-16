import numpy as np
import gym
import solvers

env = gym.make('FrozenLake8x8-v0')
env.reset()

theta = 0.01
gamma = 1
n_episodes = 1000


print("Computing optimal policy. . .")
policy = solvers.policy_iteration(env.env, theta, gamma)
print("Computing complete. Proceeding with optimal play. . .")
wins, avg_reward = solvers.play_optimally(n_episodes, env, policy)
print("Simulations complete. Result: ")
print("Number of wins: {0}; Average reward: {1}".format(wins, avg_reward))

print("\nComputing optimal policy with value iteration. . .")
value_iteration_policy = solvers.value_iteration(env.env, gamma, theta)
print("Computation complete. Proceeding with optimal play. . .")
wins, avg_reward = solvers.play_optimally(n_episodes, env, value_iteration_policy)
print("Number of wins: {0}; Average reward: {1}".format(wins, avg_reward))
