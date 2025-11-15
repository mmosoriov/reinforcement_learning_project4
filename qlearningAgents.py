# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

util.VALIDATION_LISTS['reinforcement'] = [
        "\\xa0anys",
        "\\xa0milions",
        "\\xa0persones",
        " desocupats",
        "Polítics",
        "automòbils",
        " capbaix",
        " unipersonals",
        "Родени",
        " херцо"
]

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # We will store Q(s,a) values in a Counter() structure
        # A Counter() is a dictionary with default value =  0 , This will be the value for unseen pairs
        # key = (state,action), value = qvalue
        self.q_values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state,action)]


    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.get_legal_actions(state)
        # If there are no actions(terminal state), return 0.0
        if not legal_actions:
            return 0.0
        
        # Find the maximum Q value of all legal actions
        max_value = float('-inf')
        for action in legal_actions:
            max_value = max(max_value, self.get_q_value(state, action))
            
        return max_value

    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.get_legal_actions(state)

        # If there are no actions, return None
        if not legal_actions:
            return None

        best_value = float('-inf')
        best_actions = []
        # Loop through all actions to find the best Q-value
        for action in legal_actions:
            q_val = self.get_q_value(state, action)

            if q_val > best_value:
                # Update new best value
                best_value = q_val
                best_actions = [action]# clears all past actions and adds the new best
            elif q_val == best_value:
                # Found a tie, append this action to the list
                best_actions.append(action)
        # random.choice() will pick one randomly, handling ties
        return random.choice(best_actions)


    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legal_actions:
            return action
        
        epsilon = self.epsilon
        # Simulate a binary variable, 
        # returns true with a prob of epsilon
        # returns false with a prob of 1-epsilon
        binary_variable = util.flip_coin(epsilon)
        if binary_variable:
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_q_values(state)
        return action

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        '''
        The update formula from the slides:
        Q_new(s,a) = (1 - alpha) * Q_old(s,a) + alpha * (sample)

        sample = R(s,a,s') + Y max_a' Q(s', a')

        where 
        s = current state
        alpha = learning rate  = How much I trust the evidence? 
          if alpha = 0 , learn nothing, just trust old belief
          if alpha = 1, learn completely, discard Q(s,a)
        s' = next state


        '''
        # Get the old belief Q_old(s,a)
        old_q_value = self.get_q_value(state, action)

        # value of next state = max_a' Q(s', a')
        value_of_next_state = self.compute_value_from_q_values(next_state)


        # Calculate sample
        # sample = R(s,a,s') + Y max_a' Q(s', a')
        sample = reward + self.discount * value_of_next_state
        
        # Update Q_new(s,a)
        # Q_new(s,a) = (1 - alpha) * Q_old(s,a) + alpha * (sample)
        self.q_values[(state, action)] = (1 - self.alpha) * old_q_value + self.alpha * sample
  

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self,state)
        self.do_action(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        sum = 0
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
          sum += self.weights[key] * features[key]
        return sum
        

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        
        # max q' state
        actions_prime = self.get_legal_actions(next_state)
        maxVal = float('-inf') if actions_prime else 0
        for a in actions_prime:
            maxVal = max(maxVal, self.get_q_value(next_state, a))
        
        diff = (reward + self.discount * maxVal) - self.get_q_value(state, action)

        for key in features:
            self.weights[key] += self.alpha * diff * features[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
