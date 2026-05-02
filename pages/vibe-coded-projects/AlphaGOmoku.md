---
layout: writing_page
title: AlphaGOmoku
description: A Reinforcement Learning based Gomoku agent trained from scratch, inspired by AlphaGo.
---

# AlphaGOmoku

> **A project exploring the synergy between Reinforcement Learning and AI-assisted development.**

[View on GitHub](https://github.com/L-Neur0/Alpha-GOmoku)

<div style="display: flex; flex-direction: column; gap: 1.5rem; margin: 2rem 0; align-items: center;">
  <img src="{{ '/assets/images/alphagomoku/menu.png' | relative_url }}" alt="AlphaGOmoku Menu" style="max-width: 500px; width: 100%; border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
  <img src="{{ '/assets/images/alphagomoku/gameplay.png' | relative_url }}" alt="AlphaGOmoku Gameplay" style="max-width: 500px; width: 100%; border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
</div>

## The Idea
AlphaGOmoku is a self-playing agent for Gomoku, **directly inspired by the AlphaGo/AlphaZero architecture**. The primary goal was to implement a rigorous reinforcement learning pipeline while leveraging modern AI coding assistants to handle the implementation overhead.

By using the **Gemini CLI** and other assistants to manage the frontend, UI, and boilerplate code, I was able to maintain a laser focus on the core RL logic: the Monte Carlo Tree Search (MCTS) integration and the neural network training dynamics.

## Reinforcement Learning & Training
The agent learns entirely through self-play, starting from zero knowledge of the game beyond the rules.

### The Architecture
- **Dual-Head Neural Network:** A deep CNN backbone with two heads:
  - **Policy Head:** Predicts the most promising moves.
  - **Value Head:** Evaluates the winning probability of the current state.
- **Search-Guided Learning:** MCTS acts as a policy improver. By searching ahead, it provides a "stronger" move distribution than the raw network, which then serves as the training target for the next iteration.

### Training Dynamics
The training loop follows the AlphaZero paradigm:
1. **Self-Play:** The agent plays games against itself to generate experience.
2. **Buffer:** States, MCTS search probabilities, and winners are stored.
3. **Optimization:** The network is optimized to predict both the MCTS search results and the final game outcome simultaneously.

## Developed with AI Assistants
This project served as a benchmark for AI-assisted research. The division of labor was clear:
- **AI Assistants:** Handled the React/Frontend implementation, UI styling, and data plumbing.
- **Human Researcher:** Designed the RL architecture, tuned hyperparameters, and validated the MCTS convergence.

This workflow demonstrates how researchers can now prototype complex, end-to-end systems by focusing on high-level algorithmic design while delegating the implementation details to specialized LLMs.
