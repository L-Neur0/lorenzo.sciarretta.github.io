---
layout: writing_page
permalink: /rl_index
title: Reinforcement learning notes
---

<h1>Reinforcement learning notes</h1>
<p>Read these in order: foundations first (MDP, POMDP, HMM), then tabular RL, then function approximation and policy methods, and finally model-based RL. Mainly based on <a href="https://arxiv.org/abs/2502.05244">Probabilistic AI (A. Krause)</a> and <a href="http://incompleteideas.net/book/the-book-2nd.html">Sutton &amp; Barto</a>.</p>

<ol class="entries">
  <li><span class="when">Ch. 1</span><span><a class="what" href="{{ '/rl_mdp' | relative_url }}">Markov decision processes</a><span class="detail">Finite MDPs, policies, value functions; Bellman expectation and optimality; policy and value iteration.</span></span></li>
  <li><span class="when">Ch. 2</span><span><a class="what" href="{{ '/rl_pomdp' | relative_url }}">POMDPs</a><span class="detail">Partial observability, belief states and the Bayes update, reduction to an enlarged MDP.</span></span></li>
  <li><span class="when">Ch. 3</span><span><a class="what" href="{{ '/rl_hmm' | relative_url }}">Hidden Markov models</a><span class="detail">Hidden states and emissions, connection to POMDPs.</span></span></li>
  <li><span class="when">Ch. 4</span><span><a class="what" href="{{ '/rl_tabular' | relative_url }}">Tabular RL</a><span class="detail">On/off-policy, model-based vs model-free, ε-greedy and optimism (R<sub>max</sub>), TD-learning, SARSA, Q-learning.</span></span></li>
  <li><span class="when">Ch. 5</span><span><a class="what" href="{{ '/rl_approx' | relative_url }}">RL with function approximation</a><span class="detail">DQN and Double DQN, policy gradients, REINFORCE, actor-critic, TRPO/PPO, DDPG, SAC.</span></span></li>
  <li><span class="when">Ch. 6</span><span><a class="what" href="{{ '/rl_3' | relative_url }}">Model-based approximate RL</a><span class="detail">Planning with MPC, learning dynamics and rewards, exploration with Thompson sampling and optimism.</span></span></li>
</ol>
