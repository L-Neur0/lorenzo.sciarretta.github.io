---
layout: note
permalink: /rl_pomdp
title: Partially Observed Markov Decision Processes
---
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    },
    svg: { fontCache: 'global' }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<div class="post-header">
  <div class="post-track">RL Notes · Chapter 2 · Foundations</div>
  <h1>Partially Observed Markov Decision Processes</h1>
  <div class="post-meta">When the agent only sees noisy observations of the underlying state.</div>
</div>

<div class="post-toc" markdown="1">
**Contents**
* TOC
{:toc}
</div>

In the previous chapter we focused on MDPs in a known environment, i.e. a fully observable setting. We saw that we can efficiently find an optimal policy (as long as the MDP is finite).

In this chapter we extend the Markov decision process to a partially observable setting, in which the agent only sees noisy observations $Y_t$ of its state $X_t$.

<div class="callout" markdown="1">
<div class="callout-label">Definition · POMDP</div>
A *Partially Observed Markov Decision Process* is specified by:
1. a set of states $X$;
2. a set of actions $A$;
3. transition probabilities $p(x' \mid x, a)$;
4. a reward function $r(x, a)$;
5. a set of observations $Y$;
6. observation probabilities $o(y, x) = P(Y_t = y \mid X_t = x)$.
</div>

Note that MDPs are controlled Markov chains, whereas POMDPs are controlled *[Hidden Markov Models](/rl_hmm)*.

POMDPs can be viewed as MDPs with an enlarged state space ($X + Y$). The idea is to consider an MDP whose states are *beliefs*:

$$
b_t(x) = P(X_t = x \mid y_{1:t}, a_{1:t-1}),
$$

i.e. probability distributions over the states of the POMDP. At $t = 0$ we have $b_0(x) = P(X_0 = x)$, the (deterministic) initial belief.

Given an action $a_t$, prior belief $b_t$ and a new observation, the dynamics are

$$
x_{t+1} \sim P(\cdot \mid x_t, a_t), \qquad y_{t+1} \sim P(\cdot \mid x_{t+1}),
$$

and the belief update is

$$
\begin{aligned}
b_{t+1}(x)
&= \mathbb{P}(X_{t+1} = x \mid y_{1:t+1}, a_{1:t}) \\[4pt]
&= \tfrac{1}{Z}\, \mathbb{P}(y_{t+1} \mid X_{t+1} = x)\, \mathbb{P}(X_{t+1} = x \mid y_{1:t}, a_{1:t})
&& \text{(Bayes' rule)} \\[4pt]
&= \tfrac{1}{Z}\, o(y_{t+1} \mid x)\, \mathbb{P}(X_{t+1} = x \mid y_{1:t}, a_{1:t})
&& \text{(definition of } o\text{)} \\[4pt]
&= \tfrac{1}{Z}\, o(y_{t+1} \mid x) \sum_{x' \in X} p(x \mid x', a_t)\, \mathbb{P}(X_t = x' \mid y_{1:t}, a_{1:t-1})
&& \text{(condition on } x'\text{)} \\[4pt]
&= \tfrac{1}{Z}\, o(y_{t+1} \mid x) \sum_{x' \in X} p(x \mid x', a_t)\, b_t(x')
&& \text{(definition of } b_t\text{)},
\end{aligned}
$$

where

$$
Z \;\doteq\; \sum_{x \in X} o(y_{t+1} \mid x) \sum_{x' \in X} p(x \mid x', a_t)\, b_t(x').
$$

Therefore the updated belief state is a deterministic mapping from the previous belief, depending only on the random observation $y_{t+1}$.

<div class="post-nav">
  <a class="post-nav-prev" href="./rl_mdp">MDPs</a>
  <a class="post-nav-next" href="./rl_hmm">Hidden Markov Models</a>
</div>
