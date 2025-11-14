# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Addendum:
# This code was modified by Gene Kim at University of South Florida in Fall 2025
# to make solutions not match the original UC Berkeley solutions exactly and
# align with CAI 4002 course goals regarding AI tool use in projects.

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

util.VALIDATION_LISTS['reinforcement'] = [
        "වැසි",
        " ukupnog",
        "ᓯᒪᔪ",
        " ਪ੍ਰਕਾਸ਼",
        " podmienok",
        " sėkmingai",
        "рацыі",
        " යාපාරය",
        "න්ද්"
]

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.run_value_iteration()

    def run_value_iteration(self):
        # Write value iteration code here
        R = self.mdp.get_reward

        for k in range(0, self.iterations):
            Val_k = self.values.copy()
            for s in self.mdp.get_states():
                self.update_value(s, Val_k)
                    
    def update_value(self, state, Val_k):
        Reward = self.mdp.get_reward

        maxVal = float('-inf')
        for a in self.mdp.get_possible_actions(state):
            val = 0 
            for s_prime, prob in self.mdp.get_transition_states_and_probs(state, a):
                val += prob * (Reward(state, a, s_prime) + self.discount*Val_k[s_prime]) 
            maxVal = max(maxVal, val)
        self.values[state] = maxVal if maxVal != float('-inf') else 0 # Probably could have cleaner way


    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        Reward = self.mdp.get_reward
        
        val = 0
        for s_prime, prob in self.mdp.get_transition_states_and_probs(state, action):
            val += prob * (Reward(state, action, s_prime) + self.discount*self.values[s_prime])
            
        return val
        

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Given a state, the argmax of actions from that state
        Reward = self.mdp.get_reward
        
        bestAction = (float('-inf'), None)
        for a in self.mdp.get_possible_actions(state):
            val = 0
            for s_prime, prob in self.mdp.get_transition_states_and_probs(state, a):
                val += prob * (Reward(state, a, s_prime) + self.discount*self.values[s_prime])
            bestAction = max(bestAction, (val, a), key=lambda x: x[0])
        return bestAction[1]
        

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        "Returns the policy at the state (no exploration)."
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state)
              mdp.is_terminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        states = self.mdp.get_states()
        num_states = len(states)
        for k in range(0, self.iterations):
            curState = states[k % num_states]
            self.update_value(curState, self.values)

            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def run_value_iteration(self):
        predeccesors = self.compute_predecessors()
        print(predeccesors)
        pq = util.PriorityQueue()

        for s in self.mdp.get_states():
            pass

    
    def compute_predecessors(self):
        predecessors = dict()

        for state in self.mdp.get_states():
            for action in self.mdp.get_possible_actions(state):
                for s_prime, prob in self.mdp.get_transition_states_and_probs(state, action):
                    if prob > 0:
                        # state is a predecessor of s_prime if it has a non-zero chance to reach it
                        pred = predecessors.get(s_prime)
                        if pred is None:
                            predecessors[s_prime] = set(state)
                        else:
                            predecessors[s_prime].add(state)

        return predecessors