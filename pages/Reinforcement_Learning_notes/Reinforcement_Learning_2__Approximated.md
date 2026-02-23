---
layout: default
permalink: /rl_approx
---
[Reinforcement Learning - Tabular](/rl_tabular)

```toc
```

We have seen on policy and off policy RL algorithms, but there is a problem, MDP and RL are polynomial in the number of actions |A| and in the number of states |X|. The goal is now to pass from simple tabular RL to large setting of |A| and |X|. To do so, we need to extend *model-free methods* such as TD learning and Q learning to large state and action spaces. 

The only way for doing so, is to learn **approximations** of value functions (through regression).

The first thing to do is to look the tabular RL setting we have seen as an *optimization* problem:

# 1. Tabular RL as optimization setting
We re-interpret the model-free methods from previous sections, TD and Q learning, as optimization problem, where each iteration is a single gradient update.
We look at an example for TD-learning:
$$
V^{\pi}(x) \leftarrow (1-\alpha_t)V^{\pi}(x)+(r + \gamma V^{\pi}(x'))
$$
This is basically an update rule of an optimization algo. The only thing we have to do is to parametrize the estimates of $V^{\pi}$ with some params $\theta$ and update  according to the gradient of some loss function. In particular, in a finite domain (e.g. tabular setting), we can parametrize the value function by learning a separate parameter for each state: 
$$
V^{\pi}(x; \theta) = \theta(x)
$$
Ok, now to re derive the update rule of $V^{\pi}$ as a gradient update, we need a loss function to derive, and we consider the following one:
$$
\bar{l}(\mathbf{\theta};x, r) = \frac{1}{2}(v^{\pi}(x) - \mathbf{\theta}(x))^2
$$
and using the Bellman equation:
$$
\bar{l}(\mathbf{\theta};x, r)=\frac{1}{2}\big(r + \gamma \mathbb{E}_{x'|x, \pi(x)} [v^{\pi}(x')]-\mathbf(\theta)(x)\big)^2
$$
which is the classical squared loss of the difference between the parameter and the target label we want to learn ($v^{\pi}$). Now we can view TD learning as SGD, the first issue is that we cannot compute the estimation, so as for TD learning we learn the bootstrapped estimate of $V^{\pi}$. This bootstrapping estimate $V^\pi$ is treated as if it were independent of the current estimate of the value function $\theta$.  To emphasize this, we write

$$
V^\pi(x; \theta_{\text{old}}) \approx v^\pi(x),
$$

where $\theta_{\text{old}} = \theta$ but $\theta_{\text{old}}$ is treated as a constant with respect to $\theta$. Basically Bootstrapping means to use “old” value estimates as labels. 

Secondly, the expectation is over the transition model which we are trying to avoid in model-free methods, so we use MC estimate using a single sample. Recall that this is only possible because the transitions are conditionally independent given the state-action pair.

Using these shortcuts, we can rewrite the loss function as:
$$
l(\mathbf{\theta};x, r, x') = \frac{1}{2}\big(r + \gamma \mathbf{\theta^{old}(x')} - \mathbf{\theta}(x) \big)^2
$$
We define the gradient of this loss w.r.t to $\theta (x)$ as: 
$$
\delta_{\text{TD}}
\coloneqq \nabla_\theta(x)\, \ell(\theta; x, r, x')
= \theta(x) - \bigl(r + \gamma \theta^{\text{old}}(x')\bigr).
$$
This error term is also called **temporal-difference (TD) error**, it compares the previous estimate of the value function to the bootstrapping estimate of the value function

---
# Model free
## 2. Value Function Approximation
So now that we have derived the value function update rule, it is natural to find an approximation of the parametrized $V(x; \theta)$ or $Q(x; \theta)$ functions, in order to scale to large settings. You may think of this as a regression problem where we map state(-action) pairs to a real number. Recall from the previous section that this is a strict generalization of the tabular setting, as we could use a separate parameter to learn the value function for each individual state-action pair. 

Our goal for large state-action spaces is to exploit the smoothness properties(i.e. V takes similar values in 'similar' states) of the value function to condense the representation. An easy way of doing so is **linear function approximation**. For example for Q:
$$
\hat{Q}^*(x, a; \theta) = \theta^T \phi(x, a)
$$
where $\phi$ is the hand designed feature map. [Or, a common alternative is to use a NN to learn these features, doing so is also known as *deep reinforcement learning*. ]

After observing a transition $(x, a, r, x')$, the update via gradient is (as done before for V): 
$$
\ell(\theta; x, a, r, x')
\coloneqq
\frac{1}{2}
\left(
r
+ \gamma \max_{a' \in \mathcal{A}} Q^\star(x', a'; \theta^{\text{old}})
- Q^\star(x, a; \theta)
\right)^2.
$$

The difference between the current approximation and the optimization target,

$$
\delta_{\mathrm{B}}
\coloneqq
r
+ \gamma \max_{a' \in \mathcal{A}} Q^\star(x', a'; \theta^{\text{old}})
- Q^\star(x, a; \theta),
$$

is called the *Bellman error*. Analogously to TD-learning, we obtain the gradient update,
$$
\theta \leftarrow \theta - \alpha_t \nabla_\theta \ell(\theta; x, a, r, x'),
\tag{12.13}
$$

$$
= \theta
- \alpha_t \nabla_\theta
\frac{1}{2}
\left(
r
+ \gamma \max_{a' \in \mathcal{A}} Q^\star(x', a'; \theta^{\text{old}})
- Q^\star(x, a; \theta)
\right)^2
$$

$$
= \theta + \alpha_t \delta_{\mathrm{B}} \nabla_\theta Q^\star(x, a; \theta).
\tag{12.14}
$$
and in *linear case*: 
$$
=  \delta_{\mathrm{B}}(x', \theta) \cdot \phi(x, a)
$$

 
![Screenshot 2026-01-03 at 12.36.07.png](./images/Screenshot 2026-01-03 at 12.36.07.png)
But this algorithm is rather slow!

---
## 3. Neural Fitted Q-iteration / DQN
To accelerate Q-learning with (neural net) function approximation: 
- use "experience replay", i.e. maintain a dataset D of observed transitions.
- clone network to maintain constant "target" values across episodes (cloned network is target network $Q(x', a'; \theta^{old})$ ) . 
Then the loss is:

$$
L(\theta) = \sum_{(x, a, r, x')}\left (r + \gamma \operatorname{max}_{a'} Q(x', a'; \theta^{old}) - Q(x,a;\theta)\right)^2
$$
How this is implemented exactly varies:

#### Increasing stability: **Double DQN**:
 Standard DQN Suffers from „maximization bias“, since we max over Q, which is still an estimate, we can get 'too high 'estimates, the idea is to split the max operation, so Double DQN uses current network for evaluating the argmax, i.e. selecting the best action, and then evaluate using the cloned target network. 
 $$
\mathcal{L}_{\text{DDQN}}(\theta)
=
\sum_{(x,a,r,x') \in \mathcal{D}}
\left(
r
+ \gamma
Q\!\left(x', a^\star(\theta); \theta^{\text{old}}\right)
- Q(x, a; \theta)
\right)^2,
$$

where

$$
a^\star(\theta)
=
\arg\max_{a'} Q(x', a'; \theta).
$$


The fundamental idea of Q-learning is that it chooses the next action by implicitly defining a policy via: 
$$
a_t = \operatorname{argmax}_a Q(x_t, a; \theta)
$$
but this is **intractable** for large/continuous action spaces. 

---
## 4. Policy search methods
Learning a parametrized policy (actor): $\pi(x) = \pi_{\theta}(x) = \pi(x; \theta)$. 
For episodic tasks (i.e. can reset agent) can compute expected rewards by "rollouts"(Monte Carlo forward sampling -> on policy)
The idea is to find optimal parameters through global optimization: 
$$
\theta^* = argmax_{\theta} \hat{J}^T(\theta)
$$
### 4.1. Policy Gradients
So the objective is to solve the above optimization problem. The objective is indeed to maximize:
$$
J(\theta) = \mathbb(E)_{x_{0:T}, a_{0:T}\sim \pi_{\theta}} \sum_{t = 0}^{T} \gamma^t r(x_t, a_t) = \mathbb{E}_{\tau \sim \pi_{}\theta} r(\tau)
$$
How can we obtain gradients wrt $\theta$? 
#Theorem  It holds that
$$
\nabla J(\theta)
= \nabla \mathbb{E}_{\tau \sim \pi_\theta}\!\left[r(\tau)\right]
= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[
r(\tau)\,\nabla \log \pi_\theta(\tau)
\right].
$$

**Proof.**

Recall that
$$
\nabla_\theta \log \pi_\theta(\tau)
=
\frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)}.
$$

Then,
$$
\nabla \mathbb{E}_{\tau \sim \pi_\theta} r(\tau)
=
\nabla \int \pi_\theta(\tau)\, r(\tau)\, d\tau
$$

$$
=
\int \left( \nabla \pi_\theta(\tau) \right) r(\tau)\, d\tau
$$

$$
=
\int \left( \pi_\theta(\tau)\, \nabla \log \pi_\theta(\tau) \right)
r(\tau)\, d\tau
$$

$$
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)\, \nabla \log \pi_\theta(\tau)
\right].
$$

 **Exploiting the MDP structure**

To obtain gradients for $J(\theta)$, we need to compute
$$
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)\, \nabla \log \pi_\theta(\tau)
\right].
$$

From the MDP, we have
$$
r(\tau) = \sum_{t=0}^{T} \gamma^t \, r(x_t, a_t).
$$

Moreover, the trajectory distribution factorizes as
$$
\pi_\theta(\tau)
=
p(x_0)
\prod_{t=0}^{T}
\pi(a_t \mid x_t; \theta)\,
p(x_{t+1} \mid x_t, a_t).
$$

Taking the gradient of the log-probability,
$$
\nabla_\theta \log \pi_\theta(\tau)
=
\nabla_\theta
\left(
\log p(x_0)
+ \sum_{t=0}^{T} \log \pi(a_t \mid x_t; \theta)
+ \sum_{t=0}^{T} \log p(x_{t+1} \mid x_t, a_t)
\right).
$$

Since the environment dynamics and initial-state distribution do not depend on $\theta$,
$$
\nabla_\theta \log p(x_0) = 0,
\qquad
\nabla_\theta \log p(x_{t+1} \mid x_t, a_t) = 0.
$$

Therefore,
$$
\nabla_\theta \log \pi_\theta(\tau)
=
\sum_{t=0}^{T}
\nabla_\theta \log \pi(a_t \mid x_t; \theta).
$$
Thus,
$$
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)\, \nabla \log \pi_\theta(\tau)
\right]
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)
\sum_{t=0}^{T}
\nabla \log \pi(a_t \mid x_t; \theta)
\right].
$$
However, even though the gradients obtained like this are unbiased, the experience has very **large variance**. They can reduce the variance using so-called baselines. 
>
 #Lemma it holds that
$$
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)\, \nabla \log \pi_\theta(\tau)
\right]
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\bigl(r(\tau) - b\bigr)\, \nabla \log \pi_\theta(\tau)
\right].
$$
>

**Proof**. 
$$
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)\, \nabla \log \pi_\theta(\tau)
\right]
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\bigl(r(\tau) - b\bigr)\, \nabla \log \pi_\theta(\tau)
\right].
$$
Then:
$$


\left[
\bigl(r(\tau) - b\bigr)\, \nabla \log \pi_\theta(\tau)
\right] = \mathbb{E}_{\tau \sim \pi_\theta}
\left[
r(\tau)\, \nabla \log \pi_\theta(\tau)
\right]
-
\mathbb{E}_{\tau \sim \pi_\theta}
\left[ b \nabla \log \pi_\theta(\tau)
\right].
$$
And, given $\nabla_{\theta} \log \pi_\theta(\tau) = \frac{\nabla_{\theta} \pi_\theta(\tau)}{\pi_{\theta}(\tau)}$ 
$$
\mathbb{E}_{\tau \sim \pi_\theta}
\left[ b \nabla \log \pi_\theta(\tau)
\right] =b \int \pi_\theta(\tau)\,
\frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)}\, d\tau =
$$

$$
=
b \int \nabla_\theta \pi_\theta(\tau)\, d\tau
$$

$$
=
b \, \nabla_\theta \int \pi_\theta(\tau)\, d\tau
$$

$$
=
b \, \nabla_\theta 1
=
0.
$$
Example: baseline = **Downstream return**
We can choose a state-dependent baseline:
$$
b(\tau_{0:t-1}) = \sum_{m=0}^{t-1} \gamma^mr_m
$$
==This baseline subtract the returns of all actions before time t==. Intuitively, using the baseline, the score gradient only consider downstream returns. 
Thus, we obtain the gradient estimator:
$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi} \big [\sum_{t=0}^T \gamma^tG_t \nabla log \pi(a_t|x_t;\theta) \big ]
$$
where $G_t$ is the bounded discounted payoff from time T: ==the reward to go== following action $a_t$ (t. It is also commonly called the (bounded) downstream return (or reward to go):
$$
G_t = \sum_{t'= t}^T \gamma^{t'- t}r_t
$$
Performing SGD with the score gradient estimator and downstream returns is known as the **REINFORCE** algorithm (Williams, 1992):
![Screenshot 2026-01-04 at 10.06.35.png](./images/Screenshot 2026-01-04 at 10.06.35.png)
(So we initialize the policy $\pi$ weights $\phi$... , set $G_t$ to the return at step $t$ and then update the weights)
The variance of REINFORCE can be reduced further:
Basic REINFORCE gradient estimate:
$$
\nabla_{\phi} J(\phi) = 
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\gamma^t \, G_t \, \nabla \log \pi(a_t \mid x_t; \theta)
\right]
$$
A common way for reducing the variance is via a stronger baseline, by subtracting a term $b_t$ to the downstream returns:
$$
\nabla_{\phi}J(\phi) = 
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\gamma^t \, \big(G_t - b_t(x_t)\big) \, \nabla \log \pi(a_t \mid x_t; \theta)
\right]
$$
One such example is the **mean over returns**:
$$
b_t(x_t) \doteq b = \frac{1}{T} \sum_{t'= 0}^T G_t'
$$
**Remark**: The big advantage of policy gradient methods is that they can be used in continuous action spaces, however, e.g. REINFORCE is not guaranteed to find a optimal policy, even when operating in very small domains it get stuck in local optima.

Next, we will combine value approximation techniques like Q-learning and policy gradient methods, leading to the more practical family of methods called **actor-critic methods**.

---
## 5. On-policy: Actor-Critic methods
They reduce the variance of policy gradient estimates by using ideas from value function approximation. 
IDEA: Use function approximation both to approximate value functions and to approximate policies. 

We need to introduce a new definition:
> 
#Definition (**Advantage Function**). Given a policy $\pi$, the *advantage functions* is: 
$$
A^{\pi}(x, a) \doteq q^{\pi}(x, a) - \underbrace{v^{\pi}(x)}_{=\mathbb{E_{a \sim \pi(x)}}Q^{\pi}(x, a)}
$$
> it measures the advantage of picking action $a \in \mathcal{A}$ when in state $x \in \mathcal{X}$  over simply following policy $\pi$. 

It has an important property, for the greedy action:
$$
\forall \pi, x: \quad max_a A^{\pi}(x, a) \geq 0
$$
The Bellman optimality principle is basically saying: the max is always bigger than the average, *the policy is optimal iff there is no advantages in any state*!!
$$
\pi \quad \text{optimal} \iff \forall x, a\quad A^{\pi}(x, a)\leq 0
$$
The Greedy Policy can be re stated as: $\pi_G (x) = argmax_ a Q^{\pi}(x, a) = argmax_ a A^{\pi}(x, a)$. 

We already saw how to estimate the value function, we will see that we can also do estimation *off-policy*. 

### 5.1 Policy Gradient Theorem
Remember from 4.1 how to compute gradients, with canonical MC based on policy  gradients technique called REINFORCE (we used the score trick and the score estimator, what the gradient look like is to compute the gradient of our cost function wit respect to our policy). 
So we derived the gradient estimator:
$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi} \big [\sum_{t=0}^T \gamma^t G_t \nabla log \pi(a_t|x_t;\theta) \big ]
$$
As we already mentioned, is quite slow, so we try to predict reward to go, by using ideas from value function estimation. This lead us to the the RL family of algos called *actor-critic* methods. We will state the *Policy Gradient Theorem*, base of this methods. 

#### 5.1.1 Reinterpreting score gradients
We want to rewrite this gradient to get value function back onboards, so we can approximate the value and the policy. We want to trade off bias and variance. 

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi} \big [\sum_{t=0}^T \gamma^tG_t \nabla log \pi(a_t|x_t;\theta) \big ]
$$
$$
\nabla J(\theta) = \text{lim}_{T \rightarrow \infty} \nabla_T J(\theta) =  \mathbb{E}_{\tau \sim \pi} \big [\sum_{t=0}^{\infty} \gamma^tG_t \nabla log \pi(a_t|x_t;\theta) \big ] 
$$
$$
= \sum_{t = 0}^{\infty} \mathbb{E}_{\tau \sim \pi_{\theta}} \big [ \gamma^t G_t \nabla log \pi(a_t|x_t; \theta)\big] = 
$$
Using nested expectations:
$$
= \sum_{t = 0}^{\infty} \mathbb{E}_{x_t, a_t} \big [\gamma^t \nabla log \pi(a_t|x_t; \theta) \underbrace{\mathbb{E}[G_t|x_t, a_t]}_{Q^{\pi_{\theta}} (x_t, a_t)}\big ] =
$$
The second expectation is over $(r_t, x_{t+1}, a_{t+1}, r_{t+1},...)$. 
$$
= \mathbb{E}_{x_t, a_t} \left [ \sum_{t = 0}^{\infty} \gamma^t Q^{\pi_{\theta}}(x_t, a_t) \nabla log \pi(a_t|x_t; \theta)\right ] = 
$$
We can rewrite it if we think what this expectation really is, we are going to write all this thing as a single integral, (occupancy measure):
$$
= \int \rho_{\theta}(x) \mathbb{E}_{a \sim \pi_{\theta}} \left [ Q^{\pi_{\theta}}(x, a)\nabla log \pi(a_t|x_t; \theta) \right] dx
$$
where $\rho$ is the *discounted state occupancy measure* (unnormalized state occupancy measure, not a prob distr.):
$$
\rho_{\theta}(x) \doteq \sum_{t = 0}^{\infty} \gamma \, p_{\theta}(X_t = x)
$$
(many variants of PGT, we derived the variant for infinite-horizon discounted payoffs). Now we abuse of notation, and we think:
$$
\doteq \mathbb{E}_{(x, a) \sim \pi_{\theta}} \left [ Q^{\pi_{\theta}}(x, a) \nabla log \pi(a_t|x_t; \theta) \right]
$$
And this is basically the #Theorem **Policy Gradient Theorem**. The idea is to plug in the estimate approximation of the state-value function $Q(x, a;\theta_Q)$. 

**Summary**: Actor-critic algos are combining:
- Actor (parametrized policy)
- Critic (value function approx.)
In Deep RL the approximations are done via NNs.


### 5.2 Online Actor-Critics
A list of algorithms one can derive from the main theorem above. The main ones use value function approximations (critics) with policy gradient method: basically approximating the gradient:
$$
\nabla J (\theta_{\pi}) = \mathbb{E}_{(x, a) \sim \pi_{\theta}} \left [ Q(x, a; \theta_Q) \nabla log \pi(a_t|x_t; \theta_{\pi}) \right]
$$
We need to update the critique, how? TD-learning! (online setting). If you subtract from this the expectation does not change, one example of a baseline you can subtract is the value function:
Variance reduction via baseline:
$$
\theta_\pi \leftarrow \theta_\pi
+ \eta_t
\Big[
Q(x, a; \theta_Q) - V(x; \theta_V)
\Big]
\nabla \log \pi(a \mid x; \theta_\pi)
$$
Advantage function estimate:
$$
A(x, a; \theta_A)
\triangleq
Q(x, a; \theta_Q) - V(x; \theta_V)
$$

$$
\theta_\pi \leftarrow \theta_\pi
+ \eta_t \,
A(x, a; \theta_A)\,
\nabla \log \pi(a \mid x; \theta_\pi)
$$
--> A2C algorithm.

--- 
Everything we discussed till now is very much on policy! wrt to the current policy, we are going to talk now about still on-policy, but that allows for bigger updates, and then with algorithms with off-policy. 
### 5.3 TRPO & PPO
The idea is to do to faster optimization![Screenshot 2026-01-04 at 12.34.43.png](./images/Screenshot 2026-01-04 at 12.34.43.png)

The basic idea is to build this trust region where it can reuse the data:
$$
\mathbb{E}_{\tau \sim p_{\theta}} [f(\tau)] = \int p_{\theta}(\tau)f(\tau)d\tau = \int p_{\theta}(\tau) \frac{p_{\theta}(\tau)}{p_{\theta_2}(\tau)} \, f(\tau)d\tau = 
$$
$$
= \mathbb{E}_{\tau \sim p_{\theta_2}} \left [\frac{p_{\theta}(\tau)}{p_{\theta_2}(\tau)} \, f(\tau) \right]
$$
This is basically what implement TRPO![Screenshot 2026-01-04 at 12.42.26.png](./images/Screenshot 2026-01-04 at 12.42.26.png)
[See tutorials]

It's a little bit off-policy, but still on policy. We are going on transitioning to more with off-policy methods:

### 5.4 Another approach to Policy Gradients
Aa starting point is: not with REINFORCE, but with Q-learning that was used for off policies (DQN). 
The initial motivation was intractability of $L(\theta)$ computation, where we had to compute $max_{a'} Q(x', a'; \theta^{old})$. One thing we could try to do is to use an actor, a parametrized policy, to try to learn predict that greedy action. SO we use a new NN: $\pi(x'; \theta_{\pi})$: we want to follow the greedy policy 
$$
\pi_G(x) = \operatorname{arg max}_a Q(x, a; \theta_Q)
$$
if we allow rich enough policies, this is equivalent to 
$$
\theta_{\pi}^* \in \operatorname{arg max}_{theta} \mathbb{E}_{x \sim \mu}\left [Q(x, \pi(x;\theta);\theta_Q) \right]
$$
where $\mu (x) > 0$ “explores all states”
The idea is just to apply SGD to this objective: we just need differentiable approximation of Q and $\pi$ (aka NN). 

**Computing the gradients**: given the objective

$$
\theta_\pi^* \in \arg\max_{\theta} \; \mathbb{E}_{x \sim \mu}
\left[ Q\bigl(x, \pi(x;\theta); \theta_Q \bigr) \right]
$$
From the chain rule:
$$
\nabla_{\theta_{\pi}}Q(x, \pi(x;\theta_Q);\theta_Q) = \nabla_a Q(x, a)|_{a =\pi(x;\theta_{\pi})}\underbrace{\nabla_{\theta}\pi(x;\theta)}_{Jacobian}
$$

But there is an **issue**: policy gradient methods rely on *randomized policies* for exploration. This method above uses *deterministic* policies. So how do we ensure sufficient **exploration**. 

Since method is **off-policy**, can inject additional action noise (e.g., Gaussian) to encourage exploration (akin to epsilon—greedy exploration)![Screenshot 2026-01-09 at 18.22.29.png](./images/Screenshot 2026-01-09 at 18.22.29.png)
One issue was the over confidence: overestimation bias. 

### TD3 
Twin delayed DDPG, using 2 actor critic networks, and evaluating the advantage with the smaller one. 

## 6. Randomized policies
Can we, instead of injecting random noise, ensure exploration by directly allowing randomized policies?

For the critic update:
$$
\theta_Q \leftarrow \theta_Q
-
\eta \nabla
\frac{1}{|B|}
\sum_{(x,a,r,x',y)\in B}
\left( Q(x,a;\theta_Q) - y \right)^2
$$

where

$$
y
=
r
+
\gamma
Q\!\left(
x',
\pi(x';\theta_\pi^{\text{old}}),
\theta_Q^{\text{old}}
\right).
$$
we can obtain unbiased gradient estimates by sampling from $a' \sim \pi(x', \theta^{old}_{\pi})$ . And how about the policy update step?
Recall, for det policies:  $$
\nabla_\theta Q\bigl(x, \pi(x;\theta); \theta_Q \bigr)
=
\left.
\nabla_a Q(x, a; \theta_Q)
\right|_{a = \pi(x;\theta)}
\;
\nabla_\theta \pi(x;\theta). \quad (0)
$$
If you were to compute:
$$ 
\nabla_{\theta_{\pi}} \mathbb{E}_{a \sim \pi(x;\theta_{\pi})} [Q(x, a; \theta_a)]
\quad \quad (1)
$$
So, if we suppose we use Gaussian policies = $a \sim \mathcal{N}(\mu_{\theta_{\pi}}, \Sigma_{\theta_{\pi}})$. we can reparametrize as:
$$
a = C(x;\theta_{\pi})\cdot \varepsilon + \mu (x; \theta_{\pi})
$$
for $\varepsilon \sim \mathcal{N}(0, I)$.
We plug this into (1):
$$
(1) = \nabla_{\theta_{\pi}} \mathbb{E}_{\varepsilon \sim \mathcal{N}(0;I)} \left [Q(x, C(x;\theta_{\pi})\cdot \varepsilon + \mu (x; \theta_{\pi}); \theta_a)\right]
$$

This resolving algorithm is called **SVG** (stochastic value gradients)!!!! (see Var. Inf. lecture). 

## 7. RL as Inference: Entropy-regularized RL

one way of doing this is to introduce a set of binary variables $\mathcal{O}_t \in {0, 1}$ with $\mathcal{O}_t=1$ denoting the optimal $a_t$. We define a likelihood on state $x_t$, the action, and the optimality of that corresponding random VA, and how it relates with exponentiated reward:
$$
p(o_t = 1, a_t \mid x_t) \propto \exp\!\left(\frac{1}{\lambda} r(x_t, a_t)\right),
\quad \lambda > 0
$$
the lambda is temperature param very large is very peeked. Very strongly favouor action with large rewards. 
With this setup, we can think about the conditional probability over trajectories in the underlying MDP. 
So, we condition the probability of observing a trajectory $\tau$ on  $o_t = 1$ (for all $t \in {1, \dots, T}$), i.e. $o_{1:T}$: 
$$
\underbrace{
p(x_1)\prod_{t=1}^T p(x_{t+1} \mid x_t, a_t)
}_{\text{Probability of } \tau \text{ under the dynamics}}
\;
\underbrace{
\exp\!\left( \frac{1}{\lambda} \sum_{t=1}^T r(x_t, a_t) \right)
}_{\text{Total reward along } \tau}
$$
this is a quite complicate distribution, but we can try to approximate it (using Var. inf.) with the policy we can actually implement:
Use parametrized policy as a variational posterior:
$$
\pi_\theta(a_t \mid x_t)
\;\approx\;
p(a_t \mid x_t, o_t = 1)
$$
This also induces a distribution over trajectories under $\pi_\theta(a_t \mid x_t)$:$$
\hat{p}_\theta(\tau)
=
\Bigg[
p(x_1)\prod_{t=1}^T p(x_{t+1} \mid x_t, a_t)
\Bigg]
\prod_{t=1}^T \pi_\theta(a_t \mid x_t)
$$
we can sample from this one. 
how should we pick our policy, to have those distr. close as possible? KL divergence: we can view the inference process as minimizing:
$$
\arg\min_\theta
\mathrm{KL}\!\left(
\hat{p}_\theta(\tau)
\;\middle\|\;
p(\tau \mid o_{1:T})
\right)
$$
This is equivalent to maximizing the entropy-regularized RL objective:
$$
\arg\max_\theta
\sum_{t=1}^T
\mathbb{E}_{(x_t,a_t)\sim \hat{p}_\theta(\tau)}
\Big[
r(x_t, a_t)
+
\lambda \, H\!\left[\pi_\theta(a_t \mid x_t)\right]
\Big]
$$
**Interpretation**
- The policy $\pi_\theta$ is trained to **approximate the posterior over optimal trajectories**.
- The entropy term $H[\pi_\theta(a_t \mid x_t)]$ **arises naturally from KL minimization**, rather than being added manually.
- The temperature parameter $\lambda$ controls the **reward–entropy trade-off**.

This entropy regularized approach is a natural wat to encourage exploration in MDPs:
$$
J_\lambda(\theta)
= J(\theta) + \lambda H(\pi_\theta)
$$

$$
= \mathbb{E}_{(x,a)\sim \pi_\theta}
\Big[
r(x,a) + \underbrace{\lambda H\big(\pi_\theta(\cdot \mid x)}_{\text{Use entropy of action distribution to encourage exploration}}\big)
\Big]
$$
This can suitably define regularized (action)-value functions, called *soft* value functions. 
One can derive the same previous algo (SVG), now you also have now the action entropy, so the resulting algo is **SAC: Soft Actor-Critic**, it is the same as SVG but you encourage some amount of entropy in the action distribution. This is one of the most common actor critic off policy method used. 

You can also derive a very similar approach instead of regularizing the entropy having a constraint on the KL divergence (**MPO**): updates policy by fitting it to a target soft distribution 
$$
q(a \mid x)
\propto
\pi_{\theta_{\text{old}}}(a \mid x)\exp\!\left(Q(x,a)\right)
$$
One main point, recall Var. Inf.:
From Entropy to KL Regularization, so far we have discussed

$$
\arg\max_{\theta}
\;\mathbb{E}_{(x,a)\sim \pi_\theta}
\Big[
r(x,a) + \lambda H\big(\pi_\theta(\cdot \mid x)\big)
\Big]
$$
A closely related problem
$$
\arg\max_{\theta}
\;\mathbb{E}_{(x,a)\sim \pi_\theta}
\Big[
r(x,a)
-
\lambda \,\mathrm{KL}\big(
\pi_\theta(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)
\big)
\Big]
$$
Instead of maximizing entropy directly, we **regularize the policy using a KL divergence** to a reference (pretrained) policy $\pi_{\text{ref}}$. Maintains proximity to a **reference policy** $\pi_{\text{ref}}$
