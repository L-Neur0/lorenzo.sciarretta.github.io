---
layout: default
permalink: /rl_3
---

[Reinforcement Learning 2 - Approximated](/rl_approx)
```toc
```


# Model based Approx. RL
In MN RL we want to learn the model of the environment, i.e. learn the transition probabilities and the reward function. 
We can use then the learned model for **plan** a new policy $\pi$. 

We start usually with a policy $\pi$ and some initial data $D$. The idea is to iterate through many episodes, collect data from $\pi$, estimate the $p, r$, plan a new policy based on the estimated model. 

So we cover: 
- planning;
- learning $f, r$;
- Trade off exploration and exploitation. 

## Planning 
We focus on planning in ==continuous, fully observed state spaces with non-linear transitions, without constraints==. 

### Planning with a known deterministic model
Assume we have a known deterministic model for the reward and dynamics:
$$
x_{t+1} = f(x_t, a_t)
$$
The objective is then:
$$
\max_{a_{0:\infty}}\sum_{t= 0}^{\infty} \gamma^tr(x_t, a_t) \quad s.t. \quad x_{t+1} = f(x_t, a_t)
$$
Since we are optimizing over infinity, this problem cannot be solved directly. 
### MPC: Model predictive control/RHC
**Solution**: Plan over a ==finite horizon H==, carry out the first action, then replan:
![Screenshot 2026-01-16 at 10.32.30.png](./images/Screenshot 2026-01-16 at 10.32.30.png)
So at each iteration we need to solve an optim problem. 
For ==deterministic== models $f$, $x_{\tau}$ is determined by $a_{t:\tau-1}$:
$$
x_{\tau} := x_{\tau}(a_{t:\tau-1}) = f(f(...f(f(x_t, a_t), a_{t+1})..., a_{\tau-1})
$$
So at each step we need to maximize over:
$$
J_H(a_{t:t+H-1}) := \sum_{[t= 0](t:t+H-1)} \gamma^{\tau - t}r_{\tau}(x_{\tau}(a_{t:\tau-1}), a_{\tau})
$$
**How do we optimize**: for continuous actions, we can analytically compute gradient (BPTT). But is challenging, so usually we use **heuristic global optimization methods**. 

*Example*: ==**Random shooting methods**==
Sampling approach towards global optimization of $J_H$. 
Generate ==m sets== of random samples $a_{t:t+H-1}^i$.
Then, pick the sequence $a_{t:t+H-1}^{i^*}$ that optimize: 
$$
i^* = \arg \max_{i \in {1..m}} J_H(a_{t:t+H-1}^i)
$$
**Limitation**: A common problem of finite-horizon methods is that in the setting of ==sparse rewards==, there is often no signal that can be followed.

**Remark**: If we use the value estimate $V$, then for $H=1$, maximizing $J_H$ coincides with using the ==**greedy policy**==! w.r.t. to V: 
$$
a_t := \arg \max_{a \in \mathcal{A}} \hat{J_1} = \text{greeedy policy}
$$

### MPC for stochastic transition models
For probabilistic transition models via MPC, need to optimize:
![Screenshot 2026-01-16 at 10.53.11.png](./images/Screenshot 2026-01-16 at 10.53.11.png)
==Computing this expectation exactly requires solving a high-dimensional integral==!!
#### MC trajectory sampling
One common approach: **Monte-Carlo trajectory sampling**.
![Screenshot 2026-01-16 at 10.54.41.png](./images/Screenshot 2026-01-16 at 10.54.41.png)

# Learning
Due to the Markovian structure of the MDP, observed transitions and rewards are (conditionally) independent: 
If we don’t know the dynamics & reward, can estimate them off-policy with standard supervised
learning techniques from a replay buffer (data set). 

For continuous state spaces, **learning 𝑓 and 𝑟 is basically a regression**(density estimation) problem. 
- Each experience $(x, a, r, x')$ provides a **labeled data point** $(z, y)$,  
  where $z := (x, a)$ is the input and $y := x'$ (resp. $r$) is the label.
How we can learn?
1. Learn trough MAP estimation; but compound errors.
2. Bayesian learning: we learn a distribution of $f$. 
	1. use GP inference
	2. Approximate inference: laplace, VI...

# Exploration vs Exploitation
![Screenshot 2026-01-16 at 11.06.05.png](./images/Screenshot 2026-01-16 at 11.06.05.png)
## Thomson sampling 
As for BO: ![Screenshot 2026-01-16 at 11.06.30.png](./images/Screenshot 2026-01-16 at 11.06.30.png)


## Optimism
![Screenshot 2026-01-16 at 11.07.14.png](./images/Screenshot 2026-01-16 at 11.07.14.png)
