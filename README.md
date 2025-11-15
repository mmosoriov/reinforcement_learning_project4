# reinforcement_learning_project4

Matthew West & Miguel Mateo Osorio Vela

https://github.com/mmosoriov/reinforcement_learning_project4

Engineering Process:
q1 - For question 1, we basically just followed the equations from the slides. All the values already had the equations written out, so it was only a matter of converting it to python code.

q2 - For question 2, we messed around with the values some at first. From that and just intuition, it became apparent that lowering the noise would make it more likely to take the risk as it believed it was more likely it would not fall off to the -100 reward.

q3 - For question 3, the same sort of reasoning applied as to question 2. Lower noise made it less risk averse, higher discount made it less likely to prefer value farther away. The living reward made it more or less likely to prioritize getting to the end as soon as possible.

q4 - For question 4, we separated the actual bellman equation off as a function, then just ran it with the newly updated values for each state.

q5 - Question 5 introduced the concept of asynchronus value iteration, were instead of updating all states at once, we prioritized each update, one at a time. In this questions solution we measured "correctnes", by calculating the difference between the expected value and the current value. The largest the difference, the highest priority to be updated. The use of the Priority queue in util.py was crucial, as its Min Heap implementation helped us to retrieve the largest "diff" as we entered all our inputs with a negative sign.

q6 - Question 6 marked the separation between offline planning and online learning. Instead of having all the information of the environment beforehand and planning the best moves, Q learning learns on the road by trial an error. We based our implementation in the update formula from the class slides: Q_new(s,a) = (1 - alpha) * Q_old(s,a) + alpha * (sample). Sample was calculated by adding the reward + discounted value of the best action I can do from the next state(s'). for this question it was key to use the Counter() class as it allowed us to return 0 (default value) when a (state,action) pair was not found. 

q7 - Question 7 presented the dilemma of Exploitation vs Exploration. we had to choose between taking the greedy approach, and using the best known action to get the instant reward,or take a suboptimal path(random action) to explore possible best states. Using random.choice and util.flipcoin() made this part pretty straight forward, just and if/else statement basically. 

q8 - After we both tried to find values that will learn the optimal policy after 50 iterations we resolved that is NOT POSSIBLE.

q9 - Our agent already worked for question 9, so we did not have to make any changes.

q10 - Approximate q-learning was relatively easy to implement. It also mainly consisted of just implementing the equations in python code. The most tricky part was figuring out how to iterate through the features, since they were also used as the keys to the weights.


AI Use:
Matthew - I didn't use AI on this project outside of python syntax

Miguel Mateo - I used AI as a suplemental resource to brainstorm