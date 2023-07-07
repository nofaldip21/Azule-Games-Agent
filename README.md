# Azule-Games-Agent

In this repo, I cannot show the code because the part of code have copyright issue. If you know the code behind it can contact me at nofaldi21@gmail.com

# Analysis of the problem

In Azul, the goal is to score as much as possible before the round ends. The round ends when one of the players have completed at least one full horizontal row of the wall tiles.

The Azul game have several challenges:
1. We may have more than one agents to compete.
2. We have uncertainty because we are not sure which other agents' move will effect the environment.
3. We should predict what the other agent would do to anticipate their actions.
4. We only have limited time to choose our action (1 sec).
5. We have a large state space and possible actions.

### Techniques we use
1. Reinforcement Learning - We have tried using Q-learning to achieve the goals.
    1. We try to tackle uncertainty by using reinforcement learning and make good policy for every state in the game.
    2. We can pre-train the q-table so the policy will improve for every time the agent is used.

2. Monte Carlo Tree Search 
    1. This algorithm combines the advantages of Q-learning and search.
    1. Monte Carlo technique is suitable for large state spaces as it doesn't need to traverse every possible state.

3. Breadth First Search with Minimax and Shaped Reward
    1. Minimax assumes the opponent wants to minimize our reward, hence it allows us to get the best move, if the state space is small enough to explore.
    2. This algorithm is easy to implement and troubleshoot.


## Offense

1. Reinforcement Learning do offensive by maximizing q-value and find best policy for agent itself.
2. MCTS do offensive by simulating possible future state and get action with maximize q-value.
3. BFS simulates possible future and choose the best action that lead to state with highest reward.

## Defense

1. Reinforcement learning anticipate the opponent action by making policy that is best on given state.
2. MCTS anticipates opponent action by considering opponent action as effect of environment.
3. BFS anticipates opponent by considering difference between our reward and opponent reward as heuristic.


# Reinforcement Learning (Q learning)

# Table of Contents
- [Governing Strategy Tree](#governing-strategy-tree)
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

## Governing Strategy Tree  

### Motivation  
The Azul games have uncertainty on what is opponent next action. We cannot use ordinary classical model because it returns sequence of action. Sequence of action from initial state to goal state can be fail because opponent can move differently from what we predict or expect at beginning. Hence, we propose to use reinforcement learning to solve this problem. Regardless what is the state that we will end up, we already have policy what is the best action to choose in this state by using reinforcement learning. we also can train the agent to improve policy by saving the q-table.        

[Back to top](#table-of-contents)

### Application  

- Goal: score the highest point as possible.
- Terminating state: when one of agent complete a row of end round
- The state describes:
  1. Number and color in floor tiles
  2. Number and color in wall tiles
  3. Number and color in all factory tiles
  4. Number and color in the center tile
  5. Number and color in pattern lines

- The actions are the permutation of:
  - Taking tiles from:
    - Take all tiles of the same color from the centre tile.
    - Take all tiles of the same color from one of the factory tiles.
  - Placing tiles into:
    - Place the tiles into one of the legal pattern lines and the rest to the floor tile.
    - Place all tiles on the floor tile.

- The reward function: We calculate a provisional score using `ScoreRound()` method of two agents, and computing the difference between the two.
- We do 5000 iteration. each iteration stops until reach terminate state to build Q-table before choose an action
- Before the agent class is loaded, we use 15 second to load trained Q-table

[Back to top](#table-of-contents)

### Solved Challenges

* Even though there are uncertainties in the problem, the algorithm managed to return a policy.
* The algorithm manages to update the q-table and return policy within 1 second.
* We used Pickle to persist the q-table as a file for future games.
 
   
[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  
1. We can use previous q-table to choose action in next game.
2. We can also pre train q-table until convergence.

#### *Disadvantages*
1. We have a large state space so we need the agent to train multiple time until it reach optimal solution, which could take an enormous amount of computing power.
2. We focus only on our reward without considering other agent reward and move.

[Back to top](#table-of-contents)

### Future improvements  
1. We can use multi-agent reinforcement learning
2. We need to train more games to make q-table reach optimal solution
3. We can shape reward by considering other agent action
4. Given the complexity of the state space, we can use modeling and abstraction to reduce the state space.

[Back to top](#table-of-contents)

# Multi-agent Monte Carlo Tree Search

# Table of Contents
- [Governing Strategy Tree](#governing-strategy-tree)
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

## Governing Strategy Tree  

### Motivation  

Ideally, given unlimited time, we can use Minimax to come up with the best policy that would beat the opponent 100% of the time. However, in this game the time is limited to just 1 second. Furthermore, the state space of Azul is enormous, which makes searching the entire state space impractical to do in 1 second.

Monte Carlo Tree Simulation combines best of both searching and reinforcement learning. It is able to approximate the transition function and expected future rewards without needing to search the entire state space. Hopefully, MCTS would result in a good policy that relies less on the Hiro Heuristic outlined in the BFS technique we use.

[Back to top](#table-of-contents)

### Application  

We run the tree search for a maximum of 0.9 seconds, which is 10% under the limit to give it some leeway. The action selection during the Selection step uses Upper Confidence Trees multi-armed bandit algorithm, with the exploration constant set to `1 / sqrt(2)`.

During the Expansion step, the expanded action is chosen at random from the list of unexpanded actions.

During the Simulation step, the agent simulates from the expanded state towards the end of the round. The rewards are approximated using our proprietary Hiro Heurstic we used in our BFS algorithm.

During the Backpropagation step, the score is stored as a matrix in each Node. The discount factor we used is 0.9. Note that the Q-value is not calculated directly and instead the node records its visited count and the (discounted) sum of scores. This would allow us to define the Q-function separately.

In theory, MCTS should be good for this problem. Unfortunately, this agent failed to compare with our BFS with Minimax algorithm.

[Back to top](#table-of-contents)

### Solved Challenges

- Understanding MCTS is challenging. We ended up writing 3 different MCTS implementations.
- There are multiple ways of representing the Q-functions, Q-tables, nodes, and states. We ended up using the one we think have the best separation of concerns. 

[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  
1. Unlike BFS, this is a Reinforcement Learning hence, given enough time and compute resources, this should be able to converge into an optimal solution.
2. MCTS does not look at the entire state space, therefore is suitable for this problem which has a large state space.

#### *Disadvantages*
1. Because of the time and resource constraints, MCTS still needs heuristic to guide the Expansion step. Apparently, our proprietary Hiro Heuristic did not work well for MCTS.
2. The algorithm itself is quite challenging to understand, particularly when adopting the concepts of MCTS to the problem of Azul.

[Back to top](#table-of-contents)

### Future improvements  
1. We could use better heuristic for the Expansion step to ensure "good" actions are expanded first.
2. We could fine tune the exploration constant of UCT.
3. We could employ a second Q-function to break the ties between states.
4. Given the complexity of the state space, we can use modeling and abstraction to reduce the state space.

[Back to top](#table-of-contents)

# Breadth-first Search with Modified Minimax

# Table of Contents
- [Governing Strategy Tree](#governing-strategy-tree)
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

## Governing Strategy Tree  

### Motivation  
We need method to anticipate others agent action but we have limitation of large state space and limited time to take an action. We need a simple algorithm that can simulate as many as possible action and return with the most promising action within 1 second. We propose breadth first search but we construct an heuristic that calculate the difference between scores, line tiles, floor tiles, bonus between our agent and opponent agent, which tries to evaluate how good an agent's state is. We calculate the heuristic by doing simulation on both our agent and opponent agent.


[Back to top](#table-of-contents)

### Application  

We employ breadth-first search to find the most promising action for as much depth as possible within one second. Technically, we're not searching for a specific goal state, but we traverse the state tree the same way as BFS, to calculate which action would result in the best possible future reward. 

The reward function is approximated from the current state using a heuristic we call Hiro Heuristic, named after our team member Hiro @hamanori who crafted this shaped reward heuristic. Essentially, the Hiro Heuristic take into account the following state variables:
- The agent's and opponent's provisional score (calculated using `ScoreRound()`).
- The number of tiles on the floor of our agent and opponent.
- The number of consecutive placed tiles on the wall of our agent and opponent.
- The occupancy of the pattern lines of our agent and opponent.
- Entropy of the centre and factory tiles of our agent and opponent.

Given that `getLegalActions()` returns a somewhat deterministic sequence of actions (e.g. actions regarding the first factory tiles will appear first), we randomize the actions array to ensure an even distribution of action calculation.

This algorithm roughly implements the concept of a Minimax algorithm, albeit not exactly. 
- At states with odd-numbered depth, the states represent the possible moves for our agent. The algorithm calculates the best Hiro Heuristic for the states and *adds* the heuristic value to the q-table for the original action that leads to that state.
- At states with even-numbered depth, the states represent the possible moves for our opponent's agent. The algorithm calculates the best Hiro Heuristic for the states and *subtracts* the heuristic value from the q-table for the original action that leads to that state.

This way, when simulating the states, our agent will maximize our score, and the opponent's will minimize our score.

[Back to top](#table-of-contents)

### Solved Challenges
* The algorithm manages to evaluate our agent's and other opponent's state by using the Hiro Heuristic, leading to better action.
* Even though we have a large state space, the algorithm manages to choose the most promising action, especially when the number of next legal actions is small.
* We turned BFS into an anytime algorithm like MCTS.
 
[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  

1. The algorithm is easy to implement.
2. The algorithm considers opponent action in the state.
3. It is an anytime algorithm, so given more time it would yield better actions.

#### *Disadvantages*

1. The algorithm is greedy and need time yield the best possible action.
2. On states with large number of possible moves, it will not be able to traverse deep enough to find the best possible action.
3. The algorithm assumes the opponent moves based on the same heuristic, so it does not work well if the strategies are quite different.

[Back to top](#table-of-contents)

### Future improvements  

1. We need to improve the algorithm to also do exploitation, rather than just exploration. Maybe we can employ multi-armed bandit algorithm.
2. We can memoize the visited nodes for future rounds and games.
3. The heuristic can be learned from records, which can lead to evaluate a state more properly.

[Back to top](#table-of-contents)

# Evolution of the approach

In our demo, Agent 1 is our agent and agent 0 is opponent agent

## Reinforcement Learning Approach
----

From the demo, we can see that our algorithm more focus on how to reach the terminate state and doesn't care about floor punishment. It doesn't care about bonus so the algorithm doesn't perform well on competition with other agents. It also doesn't perform very well because the algorithm need a lot of training until it converges as we have a large state space in this game.

The competition table shows the results of 10 games between reinforcement learning and other agents. From the table, we found that our implementation of reinforcement learning performs well against a random agent, but it didn't perform well against the First Move agent. This happens because our agent focuses more on its own score. Meanwhile, we also have to consider the bonus score. Furthermore, we have a large number of possible states and actions, so the Q-table does not converge to the optimal solution.

### Demo

![Demo 1](https://github.com/nofaldip21/Azule-Games-Agent/blob/master/wiki-template/images/algorithm_1.gif)

#### Competition results

| Opponent Agent | Number of Win | Number of Lost | Average Score |
| --- | --- | --- | --- |
| Random | 8 | 0 | 7.09 |
| First Move | 5 | 4 | 5.5 |
| MCTS | 7 | 2 | 4.7|
| BFS with Minmax | 0 | 10 | 1.4|

#### Strategy summary

| Pros | Cons |
| --- | --- |
| Agent can be pretrain | Agent need a lot of pre-train until converge |
| Agent doesn't need Heuristic | Agent only care how to end the game not maximize bonus |

----

## Monte Carlo Tree Search
----

Monte Carlo Tree Search doesn't perform any differently from reinforcement learning. It also focuses more on how to reach the end round, like reinforcement learning. The algorithm doesn't care about floor punishment or bonus points. Without any heuristic, the algorithm doesn't perform well because it will simulate based on random nodes.

The competition table shows that MCTS performs worse than reinforcement learning because the algorithm doesn't explore the entire state space and the chance of convergence is low.

### Demo

![Demo 2](https://github.com/nofaldip21/Azule-Games-Agent/blob/master/wiki-template/images/algorithm_2.gif)

#### Competition results

| Opponent Agent | Number of Win | Number of Lost | Average Score |
| --- | --- | --- | --- |
| Random | 2 | 8 | 2.4 |
| First Move | 2 | 5 | 1.7 |
| RL | 2 | 7 | 2.2|
| BFS with minmax | 0 | 10 | 0 |

#### Strategy summary

| Pros | Cons |
| --- | --- |
| MCTS can be pretrain | MCTS is very hard to reach convergence |
| The problem doesn't need a lot of action to get reward | Good heuristic must be construct to maximize reward |

----

## Breath First Search with Minmax
----

Breath-first search with minmax tries to get as many bonuses as possible before ending the game. As shown in the demo, the agent can finish the game right away, but it tries to make other moves that maximize the bonus in the endgame. The algorithm also considers negative rewards that come from floor punishment. Hence, the algorithm performs very well in terms of score at the end of the game.

The competition table shows a significant increase in the average score compared to other algorithms, and the algorithm can easily beat the random and first move agents. The algorithm successfully performs simulation and focuses on promising nodes by using the heuristic. The heuristic is more focused on bonus points and punishment points.

### Demo

![Demo 3](https://github.com/nofaldip21/Azule-Games-Agent/blob/master/wiki-template/images/algorithm_3.gif)

#### Competition results

| Opponent Agent | Number of Win | Number of Lost | Average Score |
| --- | --- | --- | --- |
| Random | 10 | 0 | 55.0 |
| First Move | 8 | 2 | 37.20 |
| MCTS | 10| 0 | 39 |
| RL | 10| 0 | 40 |

#### Competition Leaderboard

![Leaderboard](https://github.com/nofaldip21/Azule-Games-Agent/blob/master/wiki-template/images/Leaderboard_submission.jpg)

We only submitted using this algorithm because it performs the best compared to other algorithms. It shows good results by beating some staff agents in the competition.

#### Strategy summary

| Pros | Cons |
| --- | --- |
| The Agent consider bonus and opponent action before ending the round | The agent depends on heuristic function |
| The algorithm is easy to implement | The algorithm cannot be pretrain |




## Conclusions and Learnings

- In summary, Azul games have large state space and action. It also has limited time for the agent to determine next action. 
- Algorithms such as Q-learning and MCTS don't work well for this assignment because these algorithm need a lot of training until the q-table converge. 
- Without any heuristic, MCTS doesn't work well in simulation step because it doesn't have any guidance to pick good state. 
- Meanwhile, BFS with Minimax uses heuristic that consider different of bonus, line tiles, floor tiles, and score between agents. The algorithm works well because the heuristic consider bonus and the algorithm will construct solution where return largest bonus. 

## Reflections
- We tried numerous techniques for this assignment, rewrote the code multiple times. Most of them failed to beat a simple Breadth-first Search with the *magical* Hiro Heuristic.
  - Sometimes, the simplest solution is the best one.
  - FYI, Hiro wrote the first version of the Hiro Heuristic basically overnight.
- Doing this competition really made us think hard and understand the subject materials better. 
  - Things might be different if there is no competition. We may not have the motivation to keep pushing the performance.
  - There is a reason why our team name is Enforced Marks Climbing: Just like Enforced Hill Climbing that consistently moves uphill to reach the goal state, we want to consistently move towards a higher mark :D
- Overall, it is a fun competition that made us learn in a fun way.

