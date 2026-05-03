---
layout: note
permalink: /rl_hmm
title: Hidden Markov Models
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
  <div class="post-track">RL Notes · Chapter 3 · Foundations</div>
  <h1>Hidden Markov Models</h1>
  <div class="post-meta">Hidden states, observable emissions, and the link to POMDPs.</div>
</div>

<div class="post-toc" markdown="1">
**Contents**
* TOC
{:toc}
</div>

A *Hidden Markov Model* (HMM) is a Markovian process with **unobservable** states $X_t$ and observations $Y_t$ that depend on $X_t$ in a known way.

![Screenshot 2025-11-28 at 09.55.43.png](./images/Screenshot 2025-11-28 at 09.55.43.png)

<div class="post-nav">
  <a class="post-nav-prev" href="./rl_pomdp">POMDPs</a>
  <a class="post-nav-next" href="./rl_tabular">Reinforcement Learning — Tabular</a>
</div>
