"""
utils/normalization.py
========================
Fonctions de normalisation des observations.

Pourquoi normaliser ?
  - Le réseau de neurones de PPO (Actor et Critic) converge beaucoup
    plus vite et plus stablement quand toutes les entrées ont un
    ordre de grandeur similaire (typiquement [-1, 1]).
  - Sans normalisation, une composante en mètres (ex: distance 0-150)
    écraserait le gradient d'une composante en radians (-pi, pi),
    ce qui déséquilibre l'apprentissage des poids du réseau.
  - Cela stabilise aussi l'estimation de l'avantage (GAE) et la
    fonction de valeur du Critic.
"""

import numpy as np


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Ramène `value` de [min_val, max_val] vers [-1, 1], avec clipping."""
    normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def denormalize(value: float, min_val: float, max_val: float) -> float:
    return min_val + (value + 1.0) / 2.0 * (max_val - min_val)