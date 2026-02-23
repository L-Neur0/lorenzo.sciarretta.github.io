---
layout: default
permalink: /rl_tabular
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>

* TOC
{:toc}
"An agent navigating through MDPs". The difference from the previous chapters is that in RL the policy is randomized. Reinforcement learning is concerned with probabilistic planning in
unknown environments. 

The crucial trade-off in RL is *exploring vs exploiting*, as we have seen in BO. What does it mean? 
The agent needs to decide whether is better exploring and learning about the environment, or just exploiting its knowledge to maximize rewards. 

Indeed, BO can be seen as RL with fixed space and continuous action space: in BO we used *regret* as a performance metric (minimizing regret corresponded to maximizing reward). 

Before delving in RL definitions and common practices, we recall what is different in RL from standard supervised learning: 
- Data is not i.i.d.;
- In RL the data we get depends on our actions;
- Some actions have higher rewards than others. 
But what is data in RL? Data are collected from the environment, dynamically. So we say that they consist in *Trajectories*.

# Data

> #Definition :(Trajectory). A *Trajectory* $\tau$ is a possibly infinite sequence:

$$
\tau \doteq (\tau_0, \tau_1, \tau_2, \cdots)
$$

> of transitions $\tau_i \doteq (x_i, a_i, r_i, x_{i+1})$ where $x_i$ is the starting state,
> $a_i$ is the played action, $r_i \in \mathbb{R}$ is the attained reward and $x_{i+1}$ is the ending state.



# Settings
The collection of data is commonly classified into three settings:
1. *episodic setting*: the agent performs a sequence of “training” rounds (called
episodes), after each the env. resets. 
2. *continuous setting*: (or non-episodic): the agent learns online, leading to a single trajectory. 
3. *offline setting*: Agent must learn from fixed dataset: like a fixed "library" of pre-recorded trajectories. 

# Policy method
Another important distinction in how data is collected is:
1. *On Policy* method (learning by doing (and only from what you just did)): The agent can choose to follow a policy: it can update the policy based only on the data immediately collected using current policy only (e.g., **SARSA**, **PPO**).
2. *Off Policy* method (Learning by watching (others or your past self)): the **target policy** (the strategy the agent is trying to learn) is different from the **behavior policy** (the strategy used to actually generate the data); they are more efficient (e.g., **Q-Learning**, **DQN**).

# Model approach
There are 2 fundamental approach to RL: 
1. *Model Based RL*: aim to learn the underlying Markov decision process. We estimate(learn) the *transition probabilities*  $p$ and the reward function $r$. 
2. *Model-free RL*: they aim to learn the value function directly. 

We begin by discussing model-based methods. 


# Model based approach
We don't know how the world works, we try to "build a map" (the model) of it using the data we've collected.
## Learning the MDP
The idea is that we need to estimate the *transition probabilities* and the *reward function*: 
$$
P(X_{t+1} = x' | X_t = x, A_t = a)
$$
and 
$$
r(X = x, A = a)
$$

To determine the parameters that maximize the expected reward, we can apply Maximum Likelihood Estimation (MLE) to our observed data. 
We can think the transitions $x'|x, a$ as a sampling from a categorical random variable of which we want to estimate the success probabilities for landing in each states. 
Considering the Markovian assumption, we can think of estimating the *transition probabilities* as basically the frequency, so the sample mean: 
$$
	\hat{p}(X_{t+1}|X_t, A) = \frac{\#(X_{t+1}, X_t, A)}{\#(X_t, A)}
$$
where numerator counts the number of transitions from $x$ to to a $x'$ when playing action $a$. 
Similarly, we obtain the maximum estimation of the reward function as: 
$$
\hat{r}(x, a) = \frac{1}{\#(a|x)}\sum_{t = 0, x_t=x, a_t = a}r_t
$$
How do we handle the exploration exploitation dilemma here?
## Exploration vs Exploitation dilemma
Given this current environment, how do we choose actions?
So if we stick by exploitation we might get stuck in a suboptimal, if we keep choosing a random action, we will eventually correctly estimate all probs and rewards; but many will perform very poorly.  

So it really depends on the setting, but finding a good balance is doable with "$\varepsilon$ - greedy"

### $\varepsilon$ - Greedy
This is the simplest idea to have a good balance. 
The idea is: at each time step $t$ we throw a biased coin. If it lands head, we pick an action uniformly at random among all actions (exploration), if it lands tails, we pick the best action under the current model(exploitation). It is biased, and it is called $\varepsilon$ greedy because the prob of a coin of landing heads at time $t$ is $\varepsilon_t$. 

![Screenshot 2025-11-30 at 11.16.26.png](./images/Screenshot 2025-11-30 at 11.16.26.png)

This can be a general framework to address the exploration-exploitation dilemma.  Amazingly, this simple algorithm actually works well. 

### Optimism
Recall from the multi armed bandit problem that a key principle in effectively trading exploration and exploitation is *optimism in the face of uncertainty*. The key idea, applied to RL, is that the dynamics and reward estimation works in our favor: i.e. we will find quite good estimates of the true dynamics and rewards. 

Basically we look for *optimistic* estimates of $p, r$, from which we obtain an optimistic underlying MDP, with a bias towards exploration. In particular the rewards obtained from this MDP are upper bounds of the true MDP's rewards. 

Technically, if $r$ and $p$ are unknown, we set: 
$$
\hat{r}(x^*, a) = R_{max} \quad \forall a \in A
$$

$$
\hat{p}(x^*|x^*, a) = 1 \quad \forall a \in A
$$
Which are the maximum reward the agent can get and the probability of the *fairy-tale state* $x^*$. 

In practice, the decision of when  the estimates are *good enough* has to be tuned. 

![Screenshot 2025-11-30 at 11.40.37.png](./images/Screenshot 2025-11-30 at 11.40.37.png)

How many transitions do we need to estimates the *good enough* $p, r$ estimates? How many samples? 
We use *[Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)* to get an idea: we need to consider the transitions and the rewards as **conditionally independent** In this case the inequality tell us that for the absolute approximation error to be below $\epsilon$ with probability at least $1 - \delta$, we need:
$$
N(a|x)\geq \frac{R^2_{max}}{2\epsilon^2}\operatorname{log}\frac{2}{\delta}
$$
We get the following theorem:
> #Theorem: (Convergence of Rmax, Brafman and Tennenholtz (2002)) 
> With probability at least $1-\delta$ , $R_{max}$ reaches an $\epsilon$-optimal policy in a number of steps that is polynomial in $|X|, |A|, T, 1/\epsilon, 1/\delta$, and $R_{max}$. 

## Challenges of model based approach
1. **Memory**: We need to store all the params, so the memory required is high: that is for each $x, a, x'$ we need to store $p$ and $r$. In tabular setting is $O(n^2m)$. 
2. **Computation time**: We need to repeatedly compute the policy by solving the estimated MPD: $O(nm)$. 


# Model free approach
Model free approach estimate the value function directly. Therefore, they do not need to remember the policy optimization neither the underlying MDP. 

## On policy value-estimation
Let suppose that our agent use a fixed policy $\pi$, then we can estimate the *value funciton* as we described before as: 
$$
v^{\pi}(x) = r(x, \pi(x)) + \gamma \sum_{x' \in X} p(x'| x, \pi(x)) \cdot v^{\pi}(x')
$$

This is an estimation, a first idea is to estimate through MC, however Monte Carlo approximation does yield an unbiased estimate. And the issue is that the estimation of $v^{\pi}$ does in turn depend on the true value of $v^{\pi}$. The key idea is to use a bootstrapping estimate of the value function
instead. Due to its use in estimating the value function, bootstrapping is a core concept to model-free reinforcement learning. Crucially, using a boot-strapping estimate generally results in biased estimates of the value function. 

One way to practically compute the value function update is by using samples.
The variance of the estimate is typically reduced by mixing **new** estimates of the value function with previous estimates using a learning rate $\alpha_{t}$. This yields the temporal-difference learning algorithm: TD-Algorithm.

![Screenshot 2025-12-27 at 19.28.36.png](./images/Screenshot 2025-12-27 at 19.28.36.png)
(Will use a lookup table for the values give the state).


## SARSA
### On-policy
### Off-policy

## Off-policy value estimation

At state $x$ pick an action $a$ to obtain a transition $(x, a, r, x')$. Then update the value estimate using **bootstrapping**:
$$
\hat{Q}^{\pi}(x, a) \leftarrow (1-\alpha_t)\hat{Q}^{\pi}(x, a) + \alpha(r + \gamma\underbrace{\hat{Q}^{\pi}(x', \pi(x'))}_{\hat{V}^{\pi}(x')})
$$
As for the on-policy, the theorem of convergence becomes:
#Theorem If learning rate $\alpha_t$ satisfies: 
- $\sum_t \alpha_t = \infty$
- $\sum_t \alpha_t^2 < \infty$
And all states-action pairs are visited infinitely often, then $\hat{Q}^{\pi}$ converges to  $Q^{\pi}$ with probability 1. 

Remark: it is **off policy** because action $a$ does NOT need to be picked following $\pi$.

## Optimal value estimation
Remember that: 
1. The optimal value function is $V^*(x)$ induce the optimal policy $\pi^*$. 
2. For optimal value function it holds: 
   $V^*(x) = \operatorname{max}_a Q^*(x, a)$ where $Q^*(x, a) = r(x, a) + \gamma \sum_{x'} P(x'\ x, a) V^*(x')$. 
So the idea is to estimate $Q^*(x, a)$ directly from samples. 

Now we are going to look some of the most common ways of estimating $Q^*(x, a)$: 

### 1) Q-learning
Suppose that we have 
- an initial estimate of $\hat{Q}*(x, a)$ 
- observe the transition $x, a, x'$ with reward $r$. 
Then given the optimal value function we have: 
$$
\hat{Q}^*(x, a) \leftarrow (1-\alpha_t)\hat{Q}^*(x, a) + \alpha_t(r + \gamma \operatorname{max}_a \hat{Q}(x',a))
$$
Remember the convergence theorem: 
#Theorem If learning rate $\alpha_t$ satisfies: 
- $\sum_t \alpha_t = \infty$
- $\sum_t \alpha_t^2 < \infty$
And all states-action pairs are visited infinitely often, then $\hat{Q}^*$ converges to  $Q^*$ with probability 1. 

How do we handle the exploration-exploitation trade off? We have 
- epsilon greedy
- Optimistic exploration
#### 1.1) Convergence of optimistic Q-learning
Similar to $R_{max}$:
Initialize $\hat{Q}^*(x, a) = \frac{R_{max}}{1 - \gamma} \prod_{t =1}^{T_{init}}(1- \alpha_t)^{-1}$ at time $t$ pick: $a_t \in \operatorname{arg max}_a \hat{Q}^*(x_t, a)$.

#Theorem With probability $1 - \delta$, optimistic Q-learning obtains an $\varepsilon$-optimal policy after a number of time steps that is polynomial in $|X|, |A|, 1/\varepsilon$ and $log(1/\delta)$. 

**Properties**: 
- Memory required: we need to store the $\hat{Q}^*(x, a)$ value for each $x, a$: $O(n \cdot m)$.
- Computation time: $O(m)$ per iteration, where $\alpha_t \in \operatorname{argmax}_a \hat{Q}(x_t, a)$.

### Neural Fitted Q-iteration/DQN
To accelerate Q-learning with (neural net) function approximation: 
- use "experience replay", i.e. maintain a dataset D of observed transitions.
- clone network to maintain constant "target" values across episodes. 

$$
L(\theta) = \sum_{(x, a, r, x')}(r + \gamma \operatorname{max}_{a'} Q(x', a'; \theta^{old}) - Q(x,a;\theta))^2
$$

The fundamental idea of Q-learning is that it chooses the next action by implicitly defining a policy via: 
$$
a_t = \operatorname{argmax}_a Q(x_t, a; \theta)
$$
but this is **intractable** for large/continuous action spaces. 

## Policy search methods
Learning a parametrized policy (actor): $\pi(x) = \pi_{\theta}(x) = \pi(x; \theta)$. 
For episodic tasks (i.e. can reset agent) can compute expected rewards by "rollouts"(Monte Carlo forward sampling -> on policy)
The idea is to find optimal parameters through global optimization: 
$$
\theta^* = \operatorname{arg max}_{\theta} J(\theta) \approx \operatorname{argmax}_{\theta} \hat{J}_T(\theta)
$$

[Reinforcement Learning 2 - Approximated](/rl_approx)