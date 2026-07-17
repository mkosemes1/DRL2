"""
agent/model.py
==============
Agent PPO pour l'entraînement du drone agricole.

Réseau de neurones à double tête (acteur-critique) utilisant une
distribution normale pour la politique et une estimation de la valeur.
L'architecture est un MLP simple : input → 64 → 64 → output.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from rl_template.agent import BaseAgent


class Agent(BaseAgent):
    """Agent PPO à double tête (acteur-critique).

    L'agent prend un état (observation) et produit :
      - Une distribution normale sur l'espace d'action (politique)
      - Une estimation de la valeur de l'état (critique)

    Architecture :
        Input (obs_dim) → Linear(64) → ReLU → Linear(64) → ReLU
            → policy_head → (mean, log_std) → Normal distribution
            → value_head → scalar value
    """

    def __init__(self, n_state: int, n_action: int):
        """Initialise l'agent avec les dimensions d'entrée/sortie.

        Args:
            n_state: Dimension de l'espace d'observation.
            n_action: Dimension de l'espace d'action.
        """
        super().__init__()
        self.n_action = n_action

        # --- Réseau partagé (feature extractor) ---
        self.shared = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # --- Tête acteur (politique) ---
        # Produit la moyenne de la distribution pour chaque dimension d'action
        self.policy_mean = nn.Linear(64, n_action)
        # Log écart-type appris (paramètre non liée à la couche précédente)
        self.policy_log_std = nn.Parameter(torch.zeros(n_action))

        # --- Tête critique (valeur) ---
        self.value_head = nn.Linear(64, 1)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialise les poids du réseau avec la méthode orthogonale.

        - Couches cachées : gain=sqrt(2) pour ReLU
        - Tête acteur : gain=0.01 (distributions étroites au début)
        - Tête critique : gain=1.0 (valeurs neutres au début)
        """
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Initialisation de la tête acteur avec gain faible
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.constant_(self.policy_mean.bias, 0.0)

        # Initialisation de la tête critique
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, state):
        """Passe avant : calcule les logits de politique et la valeur.

        Accepte nativement les tenseurs torch et les tableaux numpy
        (convertis automatiquement en ``float32`` tenseur).

        Args:
            state: Tensor de forme ``(batch_size, n_state)`` ou
                ``(n_state,)``, ou tableau numpy de même forme.

        Returns:
            Tuple ``(policy_logits, value)`` où :
              - ``policy_logits`` est de forme ``(batch_size, n_action)``
              - ``value`` est de forme ``(batch_size, 1)``
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        features = self.shared(state)
        policy_logits = self.policy_mean(features)
        value = self.value_head(features)
        return policy_logits, value

    def get_distribution(self, state):
        """Construit une distribution Normale sur les actions.

        Accepte nativement les tenseurs torch et les tableaux numpy
        (convertis automatiquement en ``float32`` tenseur).

        Args:
            state: Tensor de forme ``(batch_size, n_state)`` ou
                ``(n_state,)``, ou tableau numpy de même forme.

        Returns:
            Tuple ``(distribution, value)`` où ``distribution`` est une
            instance ``torch.distributions.Normal`` paramétrée par le
            réseau, et ``value`` l'estimation de la valeur de l'état.
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        features = self.shared(state)
        mean = self.policy_mean(features)
        # Exponentielle pour garantir un écart-type positif
        std = torch.exp(self.policy_log_std.expand_as(mean))
        dist = Normal(mean, std)
        value = self.value_head(features)
        return dist, value
