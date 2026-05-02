---
layout: writing_page
title: AlphaGOmoku
description: A Reinforcement Learning based Gomoku agent trained from scratch.
---

# AlphaGOmoku

> **A weekend project built to explore the boundaries of AI-assisted development (vibe-coding) and Reinforcement Learning.**

[View on GitHub](https://github.com/L-Neur0/Alpha-GOmoku)

## The Vision
AlphaGOmoku is a self-playing agent for the game of Gomoku (Five in a Row), inspired by the AlphaZero architecture. The goal was twofold: to implement a functional Monte Carlo Tree Search (MCTS) combined with a deep neural network, and to see how far I could get in a single weekend by "vibe-coding" with the **Gemini CLI**.

## Reinforcement Learning & Training
The core of AlphaGOmoku is a self-improvement loop. Unlike traditional chess or Go engines that rely on handcrafted heuristics, this agent learns by playing against itself.

### The Architecture
- **Dual-Head Neural Network:** A shared backbone (CNN) with two heads:
  - **Policy Head:** Outputs a probability distribution over all possible moves.
  - **Value Head:** Outputs a scalar evaluating the current board position (probability of winning).
- **MCTS (Monte Carlo Tree Search):** During both training and play, MCTS uses the neural network's policy and value outputs to guide its search, effectively "looking ahead" and refining the raw network predictions.

### The Training Loop
The training is purely **unsupervised**:
1. **Self-Play:** The current best model plays thousands of games against itself.
2. **Data Collection:** Each move, the MCTS search statistics (the "improved" policy) and the final game outcome are stored.
3. **Optimization:** The neural network is trained to minimize the error between its raw policy/value and the MCTS-derived targets.

## Vibe-Coding Process
This project was an experiment in high-velocity development. By using the **Gemini CLI** as a primary coding partner, I was able to:
- Rapidly prototype the MCTS logic.
- Outsource the boilerplate of PyTorch model definitions and training loops.
- Focus entirely on the "vibes"—the high-level RL strategy and hyperparameters—while the assistant handled the implementation details.

It stands as a testament to how modern AI assistants allow researchers to transform a theoretical interest (RL training loops) into a functional, trained agent in just a few days.
