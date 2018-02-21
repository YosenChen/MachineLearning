#!/usr/bin/env python2

import numpy as np
import random
import sys

TARGET = 10.0
INIT_INPUT = 1.0
TRANSFER = lambda x: x**2
TOLERANCE = 0.1

STEP_SIZE = 0.01

# RL parameters
EPSILON = 0.2  # prob for exploration
ETA = 0.3  # learning rate
GAMMA = 0.9  # discount factor


# problem parameters
class Problem(object):
    def __init__(self, t, i0, transfer, err):
        self.t = t
        self.i0 = i0
        self.transfer = transfer  # from i to y
        self.err = err
    
    def get_output(self, i):
        return self.transfer(i)

    def get_reward(self, new_output, old_output):
        # defined as negative distance change
        return abs(self.t-old_output) - abs(self.t-new_output)
        #return -abs(self.t-new_output)

    def is_end(self, i):
        return abs(self.t-self.transfer(i)) < self.err


class PIControl(object):
    # assumption: problem.transfer is an increasing function
    def __init__(self, step, problem):
        self.step = step
        self.problem = problem
        self.current_output = problem.get_output(problem.i0)
        self.current_input = problem.i0

    def update(self):  # return 1 when is_end, else 0
        if self.problem.is_end(self.current_input):
            print("Problem is solved, last output {}".format(self.current_output))
            return 1
        action = None
        if self.current_output > self.problem.t:
            self.current_input -= self.step
            action = -self.step
        else:
            self.current_input += self.step
            action = self.step
        self.current_output = self.problem.get_output(self.current_input)
        print("Input:{}, output:{}, action:{}".format(self.current_input, self.current_output, action))
        return 0


class QFunctionApprox(object):
    # scalar version of w*phi(s, a)
    def __init__(self, eta, gamma, problem):
        self.w = 0.0
        self.eta = eta
        self.gamma = gamma
        self.problem = problem

    def __str__(self):
        return "[func approx weight:{}]".format(self.w)

    def get_value(self, state, action):
        return self.w * self.phi(state, action)

    def update(self, state, action, reward, new_state, new_actions):
        # update w
        predict_Q = self.get_value(state, action)

        Q_new_state = []
        for a in new_actions: Q_new_state.append(self.get_value(new_state, a))
        V_new_state = max(Q_new_state)
        
        target_Q = reward + self.gamma * V_new_state

        self.w -= (self.eta * (predict_Q - target_Q) * self.phi(state, action))

    def phi(self, state, action):
        # phi(s, a) = 2*1_{(t-y)*a > 0} -1
        i, y = state
        return 1.0 if (self.problem.t-y)*action > 0 else -1.0


class QBinaryTable(object):
    # simplified Q table with only 2 entries
    def __init__(self, eta, gamma, problem):
        self.Q = {True: 0, False: 0}
        self.eta = eta
        self.gamma = gamma
        self.problem = problem

    def __str__(self):
        return "[Q values:{}]".format(self.Q)

    def get_value(self, state, action):
        return self.Q[self.feature_extract(state, action)]

    def set_value(self, state, action, new_value):
        self.Q[self.feature_extract(state, action)] = new_value

    def update(self, state, action, reward, new_state, new_actions):
        # update Q(s, a)
        predict_Q = self.get_value(state, action)

        Q_new_state = []
        for a in new_actions: Q_new_state.append(self.get_value(new_state, a))
        V_new_state = max(Q_new_state)

        target_Q = reward + self.gamma * V_new_state

        self.set_value(state, action, (1-self.eta)*predict_Q + self.eta*target_Q)

    def feature_extract(self, state, action):
        # extract (s, a) as a binary signal: right or wrong direction
        i, y = state
        return (self.problem.t-y)*action >= 0


class QLearning(object):
    def __init__(self, step, epsilon, Q, problem):
        self.step = step
        self.epsilon = epsilon
        self.Q = Q
        self.problem = problem
        self.current_output = problem.get_output(problem.i0)
        self.current_input = problem.i0

    def __str__(self):
        return "QLearning with: {}".format(self.Q)

    def update(self):  # return 1 when is_end, else 0
        if self.problem.is_end(self.current_input):
            print("Problem is solved, last output {}".format(self.current_output))
            return 1
        
        current_state = (self.current_input, self.current_output)
        # decide action, epsilon greedy
        action = None
        if np.random.uniform() < self.epsilon:
            # exploration
            action = self.step * (2*random.randint(0, 1) - 1)
        else:
            # exploitation
            Q_value = lambda x: self.Q.get_value(current_state, x)
            action = self.step if Q_value(self.step) > Q_value(-self.step) else -self.step
        
        # interact with the problem
        new_input = self.current_input + action
        new_output = self.problem.get_output(new_input)
        reward = self.problem.get_reward(new_output, self.current_output)

        # incorporate feedback, update Q function
        new_state = (new_input, new_output)
        new_actions = [self.step, -self.step]
        self.Q.update(current_state, action, reward, new_state, new_actions)
        
        self.current_input, self.current_output = new_state
        print("Input:{}, output:{}, action:{}".format(self.current_input, self.current_output, action))
        print("Q info:{}".format(self.Q))
        return 0


def run(control_method, iter_max = 100000):
    print(control_method)
    res = 0
    i = 0
    while res != 1 and i < iter_max:
        i += 1
        print("iteration#{}".format(i))
        res = control_method.update()

def main():
    problem = Problem(TARGET, INIT_INPUT, TRANSFER, TOLERANCE)
    pi_control = PIControl(STEP_SIZE, problem)
    run(pi_control)

    q_func_approx = QFunctionApprox(ETA, GAMMA, problem)
    q_learning_func_approx = QLearning(STEP_SIZE, EPSILON, q_func_approx, problem)
    run(q_learning_func_approx)

    q_binary = QBinaryTable(ETA, GAMMA, problem)
    q_learning_binary = QLearning(STEP_SIZE, EPSILON, q_binary, problem)
    run(q_learning_binary)

if __name__ == "__main__":
    main()
