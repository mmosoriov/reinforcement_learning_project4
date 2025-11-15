# reinforcement_learning_project4

Matthew West & Miguel Mateo Osorio Vela

https://github.com/mmosoriov/reinforcement_learning_project4

Engineering Process:
q1 - For question 1, we basically just followed the equations from the slides. All the values already had the equations written out, so it was only a matter of converting it to python code.
q2 - For question 2, we messed around with the values some at first. From that and just intuition, it became apparent that lowering the noise would make it more likely to take the risk as it believed it was more likely it would not fall off to the -100 reward.
q3 - For question 3, the same sort of reasoning applied as to question 2. Lower noise made it less risk averse, higher discount made it less likely to prefer value farther away. The living reward made it more or less likely to prioritize getting to the end as soon as possible.
q4 - For question 4, we seperated the actual bellman equation off as a function, then just ran it with the newly updated values for each state. 
q5 - 
q6 - 
q7 - 
q8 - 
q9 - Our agent already worked for question 9, so we did not have to make any changes.
q10 - Approximate q-learning was relatively easy to implement. It also mainly consisted of just implementing the equations in python code. The most tricky part was figuring out how to iterate through the features, since they were also used as the keys to the weights.


AI Use:
Matthew - I didn't use AI on this project outside of python syntax

Mateo - 