---
layout: default
permalink: /rl_pomdp
---
```toc
```
### Partially Observed Markovian Decision Process (POMDP)
So in the previous chapter we focused on MDP in a known environment, i.e. a full observable setting. We have seen that we can efficiently find the optimal policy (as long as the Markov decision process is finite).

In this chapter we are going to consider how Markovian decision process can be extended to a partially observable setting where the agent can only see noisy observations $Y_t$ of its state $X_t$. 

> #Definition : (POMDP) Partially observed Markov decision process is specified by: 
> 1. A set of states $X$;
> 2. A set of actions $A$;
> 3. Transition probabilities $p(x'| x, a)$;
> 4. a reward function $r(x, a)$;
> 5. a set of observations $Y$;
> 6. observation probabilities: $o(y, x) = P(Y_t = y | X_t = x)$. 

It is important to notice that MDP where controlled Markov chains, whether POMDP are controlled *[Hidden Markov Models](/rl_hmm)*.  

POMDPs can be considered as MDP with an enlarged state space ( $X +Y$). The idea is to consider an MDP whose states are *beliefs* : 
$$
b_t(x) = P(X_t = x| y_{1:t}, a_{1:t-1})
$$
about the current state of the POMDP. I.o.w. the states of the MDP are probability distributions over the states of the POMDP. 

At $t=0$ we have $b_0(x) = P(X_0 = x)$, this is the initial determinist state. 
At $t = t+1$ the update is given by:
Given an action $a_t$, the prior belief $b_t$ and the new observations: 
$$ 
x_{t+1}~P(\cdot| x_t, a_t) \quad \text{observe: } y_{t+1}~P(\cdot|x_{t+1})
$$
we can describe the update as: 
$$
\begin{aligned}
b_{t+1}(x)
&= \mathbb{P}(X_{t+1} = x \mid y_{1:t+1}, a_{1:t}) \\[4pt]
&= \frac{1}{Z} \mathbb{P}(y_{t+1} \mid X_{t+1} = x)\, \mathbb{P}(X_{t+1} = x \mid y_{1:t}, a_{1:t}) 
&& \text{by the definition of beliefs (10.42)} \\[4pt]
&= \frac{1}{Z} \, o(y_{t+1}\mid x)\, \mathbb{P}(X_{t+1} = x \mid y_{1:t}, a_{1:t})
&& \text{using Bayes' rule (1.45)} \\[4pt]
&= \frac{1}{Z} \, o(y_{t+1}\mid x) \sum_{x' \in X} p(x \mid x', a_t)\, 
\mathbb{P}(X_t = x' \mid y_{1:t}, a_{1:t-1})
&& \text{using the definition of observation probabilities (10.39)} \\[-2pt]
&&& \text{and conditioning on the previous state } x' \\[4pt]
&= \frac{1}{Z} \, o(y_{t+1}\mid x) \sum_{x' \in X} p(x \mid x', a_t)\, b_t(x')
&& \text{using the definition of beliefs (10.42)}.
\end{aligned}
\]

$$

where: 

$$
Z \doteq \sum_{x \in X} o(y_{t+1} \mid x) 
\sum_{x' \in X} p(x \mid x', a_t)\, b_t(x').


$$

Therefore, the update belief state is a deterministic mapping from the previous belief state depending only on the random observation $y_{t+1}$. 

->>> [Reinforcement Learning - Tabular](/rl_tabular)
