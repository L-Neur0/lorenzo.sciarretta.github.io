---
layout: writing_page
permalink: /page2
---

<div class="notes-hero">
  <h1>ML / DL / RL — Notes</h1>
  <p>Personal notes condensed from coursework and reading. Two tracks: a structured walk-through of <strong>Reinforcement Learning</strong>, and a topic-by-topic series on <strong>Statistical &amp; Machine Learning</strong>.</p>
</div>

<div class="section-header">
  <h2>Reinforcement Learning</h2>
  <div class="section-divider"></div>
</div>

<p style="color:#8b949e; font-size:0.92rem; line-height:1.6; margin: 0 0 18px;">
A linear progression — read in order. Foundations first (MDP / POMDP / HMM), then tabular RL, then function approximation and policy methods, and finally model-based RL. Mainly inspired by <a href="https://arxiv.org/abs/2502.05244">Probabilistic AI (A. Krause)</a> and <a href="http://incompleteideas.net/book/the-book-2nd.html">Sutton &amp; Barto</a>.
</p>

<div class="chapter-track">

  <div class="chapter-card">
    <div class="chapter-num">Chapter 1 · Foundations</div>
    <h3>Markov Decision Processes</h3>
    <ul>
      <li>Finite MDPs, policies, value functions</li>
      <li>Bellman expectation &amp; optimality</li>
      <li>Policy &amp; value iteration</li>
    </ul>
    <span class="chapter-link">Read chapter →</span>
    <a class="chapter-cover" href="./rl_mdp" aria-label="MDPs"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Chapter 2 · Foundations</div>
    <h3>POMDPs</h3>
    <ul>
      <li>Partial observability, observations</li>
      <li>Belief states &amp; the Bayes update</li>
      <li>Reduction to an enlarged MDP</li>
    </ul>
    <span class="chapter-link">Read chapter →</span>
    <a class="chapter-cover" href="./rl_pomdp" aria-label="POMDPs"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Chapter 3 · Foundations</div>
    <h3>Hidden Markov Models</h3>
    <ul>
      <li>Hidden states &amp; emissions</li>
      <li>Connection to POMDPs</li>
    </ul>
    <span class="chapter-link">Read chapter →</span>
    <a class="chapter-cover" href="./rl_hmm" aria-label="HMM"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Chapter 4 · Tabular RL</div>
    <h3>Reinforcement Learning — Tabular</h3>
    <ul>
      <li>Trajectories, on/off-policy, model-based vs free</li>
      <li>ε-greedy &amp; optimism (R<sub>max</sub>)</li>
      <li>TD-learning, SARSA, Q-learning, DQN</li>
    </ul>
    <span class="chapter-link">Read chapter →</span>
    <a class="chapter-cover" href="./rl_tabular" aria-label="Tabular RL"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Chapter 5 · Approximate RL</div>
    <h3>RL with Function Approximation</h3>
    <ul>
      <li>TD / Q-learning as SGD, Bellman error</li>
      <li>DQN, Double DQN</li>
      <li>Policy gradients, REINFORCE, baselines</li>
      <li>Actor-Critic, TRPO/PPO, DDPG, SAC</li>
    </ul>
    <span class="chapter-link">Read chapter →</span>
    <a class="chapter-cover" href="./rl_approx" aria-label="Approximate RL"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Chapter 6 · Model-Based</div>
    <h3>Model-Based Approximate RL</h3>
    <ul>
      <li>Planning: MPC, random shooting</li>
      <li>Learning <em>f</em>, <em>r</em> (MAP / Bayesian)</li>
      <li>Exploration: Thompson sampling, optimism</li>
    </ul>
    <span class="chapter-link">Read chapter →</span>
    <a class="chapter-cover" href="./rl_3" aria-label="Model-based RL"></a>
  </div>

</div>

<div class="chapter-flow">
  <span>Foundations</span>·<span>Tabular</span>·<span>Approximate</span>·<span>Model-Based</span>
</div>

<div class="section-header">
  <h2>LLMs</h2>
  <div class="section-divider"></div>
</div>

<div class="chapter-track">
  <div class="chapter-card">
    <div class="chapter-num">Project</div>
    <h3>Local RAG implementation</h3>
    <ul>
      <li>Retrieval-Augmented Generation, fully local</li>
      <li>Vector search + open-weight LLMs</li>
    </ul>
    <span class="chapter-link">View on GitHub →</span>
    <a class="chapter-cover" href="https://github.com/L-Neur0/Simple-Local-RAG" target="_blank" rel="noopener" aria-label="Local RAG"></a>
  </div>
</div>

<div class="section-header">
  <h2>Statistical &amp; Machine Learning</h2>
  <div class="section-divider"></div>
</div>

<p style="color:#8b949e; font-size:0.92rem; line-height:1.6; margin: 0 0 18px;">
Topic-by-topic notes covering classical ML — regression, classification, regularisation, dimensionality reduction, kernel methods, unsupervised learning, and neural networks.
</p>

<div class="chapter-track">

  <div class="chapter-card">
    <div class="chapter-num">Part 1 · Regression</div>
    <h3>Linear Regression</h3>
    <ul>
      <li>Simple &amp; multiple linear regression</li>
      <li>Model assessment, intervals</li>
      <li>Categorical / correlated predictors</li>
      <li>Polynomial regression</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog1" aria-label="Linear Regression"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 2 · Classification</div>
    <h3>Classification</h3>
    <ul>
      <li>Logistic regression</li>
      <li>Generative classifiers</li>
      <li>Evaluation of classification models</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog2" aria-label="Classification"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 3 · Validation</div>
    <h3>Resampling</h3>
    <ul>
      <li>Resampling methods &amp; cross-validation</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog3" aria-label="Resampling"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 4 · Regularisation</div>
    <h3>Model Selection &amp; Regularisation</h3>
    <ul>
      <li>Selection criteria</li>
      <li>Ridge regression</li>
      <li>Lasso</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog4" aria-label="Regularisation"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 5 · Dim. Reduction</div>
    <h3>Dimension Reduction</h3>
    <ul>
      <li>Principal Components Regression</li>
      <li>Partial Least Squares</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog5" aria-label="Dim Reduction"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 6 · Kernels</div>
    <h3>Support Vector Machines</h3>
    <ul>
      <li>Support Vector Classifier</li>
      <li>SVM with kernels</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog6" aria-label="SVM"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 7 · Unsupervised</div>
    <h3>Unsupervised Learning</h3>
    <ul>
      <li>PCA</li>
      <li>Clustering</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog7" aria-label="Unsupervised"></a>
  </div>

  <div class="chapter-card">
    <div class="chapter-num">Part 8 · Deep Learning</div>
    <h3>Deep Learning</h3>
    <ul>
      <li>Neural Networks</li>
    </ul>
    <span class="chapter-link">Read →</span>
    <a class="chapter-cover" href="./blog8" aria-label="Deep Learning"></a>
  </div>

</div>
