---
layout: default
permalink: /rl_mdp
---
```toc
```

A Markovian Decision Process, MDP, is a mathematical model for sequential decision making under uncertainty. 

The fromal definition of a **finite MDP** is:

> #Definition: A *finite MDP* is is specified by:
> 	1. a finite set of states $X$;
> 	2. a finite set of actions $A$;
> 	3. Initial state distribution $P_0$;
> 	4. *transition probabilities* $P(x'|x, a)$;
> 	5. a *reward function*  $r(x, a)$. 


In MDP setting we suppose that we know $P$ and $r$, this is why we say that it is a **known environment** , that is fully observable. 

**Goal**: the current goal is to learn how the agent should behave to optimize its reward. How do we optimally decide what is the next action? This decision map is called *policy*:

> #Definition : A *policy* is a function that maps each state $x \in X$ of the MDP to a probability distribution  over the actions: 

$$
\pi(a|x) = P(A_t = a| X_t = x)
$$

>with $t>0$. 

For MDP policies are deterministic maps. Interestingly, a *policy* introduce a **Markov Chain** ${X^{\pi}_t}_{t \in \mathbb{N}_0}$ , with transition probabilities: 
$$ 
p^{(\pi)}(x'|x) = P(X^{\pi}_{t+1} = x'|X^{\pi}_{t} = x) = \sum_{a \in A} \pi(a|x)p(x'|x, a)
$$
So if the agent follows a fixed policy, then the evolution of the process is described fully by a Markov Chain. 

But as we mentioned at the beginning the goal is to maximize the reward, how do we do that? 
There are many ways, if we consider that we know the MDP, i.e. we can evaluate it, $\forall a, x'$ , a very common way of doing so is with the *discounted payoff*:

The goal is to maximize: $\operatorname{max: }\mathbb{E}[\sum_{t=0}^{T-1} r(x_t, a_t)]$ and we discount it by a factor called *discount factor* $\gamma \in [0, 1)$ , to obtain the discounted payoff from time $t$, defined as a random variable $G_t$: 
$$
G_t = \sum^{\infty}_{m=0} \gamma^{m}R_{t+m}
$$
This is also called **Infinite horizon discounted reward**. 

Now we want to understand the effect of our initial state and initial action on the optimization objective $G_t$, to do so we use 2 functions: 
1. **State value function**: expected return starting in $x$ and following $\pi$: $v^{\pi}_t(x) = \mathbb{E}_{\pi}[G_t|X_t = x]$;
2. **State-action value function**: expected return starting in $x$, taking action $a$ and then following $\pi$:  $q^{\pi}_t(x, a) = \mathbb{E}_{\pi}[G_t|X_t = x, A_t = a]$;


So we stated that the policy is deterministic, what if it is a *randomized policy*? 
$$ \pi(x): X \rightarrow P(A)
$$
## Computing the value of a policy: 'Bellman Expectation Equation'
As we stated, a policy is associated with its *value function*, how we can compute it? 
Recall: 
$$ J(\pi) = \mathbb{E}[r(x_0, \pi(x_0) + \gamma r(x_1, \pi(x_1)) + \gamma^2r(x_2, \pi(x_2)), \cdots]
$$
and we can say: 
$$
\begin{align}
V^{\pi}(x) &= J(\pi \mid X_0 = x) \\
          &= \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r(x_t, \pi(x_t)) \mid X_0 = 1\right] \\
          &= \cdots \\
          &= r(x, \pi(x)) + \gamma \sum_{x'} p(x' \mid x, \pi(x)) \cdot V^{\pi}(x')
\end{align}
$$
This equation is called **Bellman Expectation Equation**, and it shows the recursive dependence of the value function on itself. The meaning is: the value of the current state depends on the reward of the next action from the current state + a discounted sum of all future rewards obtained from the subsequent states, given the current one. 

We can write the *Bellman expectation equation* in vector form: 
$$ 
V^{\pi} = (I - \gamma P^{\pi})^{-1}r^{\pi}
$$
So this is a linear system, and if we solve it, we obtain the value function matrix, but since it involves matrix inversion, it is cubic in time. 
Luckily there are faster way to compute this: 
### Fixed-point Iteration
Obtain an (approximate) solution of $V^{\pi}$.

*see notes and proof!*

## Policy optimization 

Recall that our goal was to find an optimal policy, such as: 
$$
\pi^*= \operatorname{argmax}_{\pi} \mathbb{E}_{\pi}[G_0]
$$
We can alternatively characterize an optimal policy as follows:
$$
\pi \geq \pi' \iff v^{\pi}(x) \geq v^{\pi'}(x), \quad \forall x \in X
$$
from this follows that  all optimal policies have **identical** value functions, we can call $v^* = v^{\pi^*}$ and $q^* = q^{\pi^*}$ the value and the action function arising from an optimal policy. Then from above we can state: 
$$
v^* = \operatorname{max}_{\pi} v^{\pi}(x)
$$
$$
q^* = \operatorname{max}_{\pi}q^{\pi}(x, a)
$$
We define the *==greedy policy==* w.r.t a value function as, for state-action function:
$$
\pi_q(x) = \arg\max_{a \in A} q(x, a)
$$


## Bellman Optimality Equation
Observe that following the *greedy policy* $\pi_v$, will lead us to a new value function of $v^{\pi_{v}}$  . The correspondence between greedy policies and value functions induces a cyclic dependency: 
![greedy_policy_value_dependency.png](./images/greedy_policy_value_dependency.png)
Notably, it turns out that the optimal policy, $\pi^*$ is a fixed-point of this dependency. So the following theorem: 

> #Theorem : (*Bellman Theorem*). A policy $\pi^*$ is optimal iff it is greedy with respect to its own value function . I. o. w., $\pi^*$ is optimal iff $\pi^*(x)$ is a distribution over the set $\operatorname{argmax}_{a \in A} q^*(x, a)$. 

From this follows: 
> ***Corollary***: The optimal value function $v^*$ is a fixed point of the so called *Bellman update*:
> $$
> v^{\star}(x) = \max_{a \in A} \big[\, r(x,a) + \gamma \, \mathbb{E}_{x' \mid x,a}[\, v^{\star}(x') \,] \big]
> $$

### Policy Iteration

*see notes*
Using above we can estimate faster than fixed point Iteration. 
### Value Iteration





 ->>> [Partially Observed Markovian Decision Process (POMDP)](/rl_pomdp)
