import numpy as np


def play_optimally(n_episodes, env, policy):
    wins = 0
    total_reward = 0
    for _ in range(n_episodes):
        terminated = False
        state = env.reset()

        while not terminated:
            action = np.argmax(policy[state])
            next_state, reward, terminated, info = env.step(action)
            #env.render()
            total_reward += reward
            state = next_state

            if terminated and reward == 1:
                wins += 1

    avg_reward = total_reward / n_episodes
    return wins, avg_reward


def one_step_lookahead(state, env, gamma, V):
    action_values = np.zeros(env.nA)

    for action in range(env.nA):
        for prob, next_state, reward, info in env.P[state][action]:
            action_values[action] += reward + (prob * gamma * V[next_state])

    return action_values


def policy_iteration(env, theta, gamma):
    policy = np.ones([env.nS, env.nA]) / env.nA


    policy_stable = False

    while not policy_stable:
        V = policy_evaluation(policy, env, gamma, theta)
        policy_stable, policy = policy_improvement(env, policy, V, gamma)

    return policy


def policy_evaluation(policy, env, gamma, theta):
    V = [0 for _ in range(env.nS)]
    while True:
        delta = 0
        for state in range(env.nS):
            v_new = 0

            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, info in env.P[state][action]:
                    v_new += action_prob * (reward + (prob * gamma * V[next_state]))

            delta = max(delta, abs(v_new - V[state]))
            V[state] = v_new

        if delta < theta:
            return V


def policy_improvement(env, policy, V, gamma):
    policy_stable = True

    for state in range(env.nS):
        old_action = np.argmax(policy[state])
        action_values = one_step_lookahead(state, env, gamma, V)
        best_action = np.argmax(action_values)

        if old_action != best_action:
            policy_stable = False

        to_view = np.eye(env.nA)[best_action]
        policy[state] = to_view

    return policy_stable, policy


def value_iteration(env, gamma, theta):
    V = [0 for _ in range(env.nS)]

    while True:
        delta = 0
        for state in range(env.nS):
            action_values = one_step_lookahead(state, env, gamma, V)
            best_action_value = np.max(action_values)
            delta = max(delta, abs(V[state] - best_action_value))

            V[state] = best_action_value

        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])

    for state in range(env.nS):
        action_value = one_step_lookahead(state, env, gamma, V)
        best_action = np.argmax(action_value)

        policy[state, best_action] = 1

    return policy