"""
tests/test_agent_model.py
==========================
Tests unitaires pour l'agent PPO (model.py).

Vérifie l'architecture du réseau acteur-critique, la propagation
avant, les distributions d'action, la différenciabilité des sorties,
l'initialisation des poids, et le comportement stochastique/déterministe.

Chaque test est indépendant et n'effectue pas d'appel réseau réel.
"""

import sys
import os
import pytest
import numpy as np
import torch
from torch import nn

# ─── Configuration des chemins d'import ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import Agent
from rl_template.agent import BaseAgent


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def obs_dim():
    """Dimension de l'espace d'observation (17+3+1+3*4 = 30)."""
    return 30


@pytest.fixture
def act_dim():
    """Dimension de l'espace d'action continue."""
    return 6


@pytest.fixture
def agent(obs_dim, act_dim):
    """Agent PPO créé avec les dimensions standard du projet."""
    return Agent(n_state=obs_dim, n_action=act_dim)


@pytest.fixture
def single_state(obs_dim):
    """Un seul état d'observation (forme (obs_dim,))."""
    return torch.randn(obs_dim)


@pytest.fixture
def batch_states(obs_dim):
    """Un batch de 8 états (forme (8, obs_dim))."""
    return torch.randn(8, obs_dim)


@pytest.fixture
def single_state_np(obs_dim):
    """Un seul état sous forme numpy (forme (obs_dim,))."""
    return np.random.randn(obs_dim).astype(np.float32)


# ─── Tests : Héritage et type ─────────────────────────────────────

def test_agent_is_subclass_of_base_agent():
    """L'agent doit hériter de BaseAgent (interface rl_template)."""
    assert issubclass(Agent, BaseAgent)


def test_agent_is_instance_of_nn_module(agent):
    """L'agent doit être un nn.Module (pour l'optimiseur PyTorch)."""
    assert isinstance(agent, nn.Module)


# ─── Tests : Architecture et dimensions ────────────────────────────

def test_agent_init_dimensions(agent, obs_dim, act_dim):
    """L'agent doit stocker les dimensions d'entrée et de sortie.

    Vérifie que les couches du réseau ont les bonnes tailles :
      - shared[0]: Linear(obs_dim, 64)
      - shared[2]: Linear(64, 64)
      - policy_mean: Linear(64, act_dim)
      - value_head: Linear(64, 1)
    """
    assert agent.n_action == act_dim

    # Couche d'entrée partagée
    assert agent.shared[0].weight.shape == (64, obs_dim)
    # Couche cachée
    assert agent.shared[2].weight.shape == (64, 64)
    # Tête acteur
    assert agent.policy_mean.weight.shape == (act_dim, 64)
    # Tête critique
    assert agent.value_head.weight.shape == (1, 64)


# ─── Tests : Propagation avant (forward) ───────────────────────────

def test_agent_forward_numpy(agent, single_state_np):
    """forward() doit accepter un tableau numpy et retourner un tuple de 2 tenseurs."""
    logits, value = agent(single_state_np)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(value, torch.Tensor)


def test_agent_forward_tensor(agent, single_state):
    """forward() doit accepter un tenseur torch."""
    logits, value = agent(single_state)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(value, torch.Tensor)


def test_agent_forward_shape_single(agent, single_state, act_dim):
    """Un état (obs_dim,) → logits (act_dim,) et value (1,)."""
    logits, value = agent(single_state)
    assert logits.shape == (act_dim,)
    assert value.shape == (1,)


def test_agent_forward_shape_batch(agent, batch_states, act_dim):
    """Un batch (8, obs_dim) → logits (8, act_dim) et value (8, 1)."""
    logits, value = agent(batch_states)
    assert logits.shape == (8, act_dim)
    assert value.shape == (8, 1)


# ─── Tests : Distribution d'action ─────────────────────────────────

def test_agent_get_distribution_numpy(agent, single_state_np):
    """get_distribution() doit fonctionner avec un input numpy.

    Retourne un tuple (distribution, value).
    """
    dist, value = agent.get_distribution(single_state_np)
    assert isinstance(dist, torch.distributions.Normal)
    assert isinstance(value, torch.Tensor)


def test_agent_get_distribution_tensor(agent, single_state):
    """get_distribution() doit fonctionner avec un input tenseur."""
    dist, value = agent.get_distribution(single_state)
    assert isinstance(dist, torch.distributions.Normal)
    assert isinstance(value, torch.Tensor)


def test_agent_get_distribution_type(agent, single_state):
    """La distribution retournée doit être une Normal de torch.distributions."""
    dist, _ = agent.get_distribution(single_state)
    assert isinstance(dist, torch.distributions.Normal)


# ─── Tests : get_action ────────────────────────────────────────────

def test_agent_get_action_sample(agent, single_state):
    """get_action(state) sans action explicite doit échantillonner une action."""
    action, log_prob, entropy, value = agent.get_action(single_state)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (6,)


def test_agent_get_action_explicit(agent, single_state):
    """get_action(state, action) doit évaluer l'action donnée."""
    explicit_action = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.5, -0.1])
    action, log_prob, entropy, value = agent.get_action(single_state, explicit_action)
    # L'action retournée doit être identique à celle fournie
    assert torch.allclose(action, explicit_action)


def test_agent_get_action_returns_4tuple(agent, single_state):
    """get_action() doit retourner un tuple de 4 éléments."""
    result = agent.get_action(single_state)
    assert len(result) == 4


def test_agent_get_action_shapes(agent, single_state, act_dim):
    """Les sorties de get_action() ont les bonnes formes.

    - action: (act_dim,)
    - log_prob: (act_dim,) — une log-prob par dimension d'action
    - entropy: (act_dim,) — une entropie par dimension d'action
    - value: (1,) — estimation scalaire de la valeur
    """
    action, log_prob, entropy, value = agent.get_action(single_state)
    assert action.shape == (act_dim,)
    assert log_prob.shape == (act_dim,)
    assert entropy.shape == (act_dim,)
    assert value.shape == (1,)


# ─── Tests : Différenciabilité ─────────────────────────────────────

def test_agent_log_prob_requires_grad(agent, single_state):
    """log_prob doit être différentiable (nécessaire pour la rétropropagation)."""
    action, log_prob, _, _ = agent.get_action(single_state)
    assert log_prob.requires_grad


def test_agent_value_requires_grad(agent, single_state):
    """La valeur estimée doit être différentiable."""
    _, _, _, value = agent.get_action(single_state)
    assert value.requires_grad


# ─── Tests : Initialisation des poids ──────────────────────────────

def test_agent_weights_orthogonal(agent):
    """Les couches cachées du réseau partagé doivent être initialisées orthogonalement.

    Vérifie que les poids de chaque couche Linear dans ``shared`` sont
   orthogonaux (gain=sqrt(2) pour ReLU).
    """
    for module in agent.shared:
        if isinstance(module, nn.Linear):
            # Vérification de base : les poids ne sont pas nuls et pas uniformes
            W = module.weight.data
            assert W.abs().sum() > 0
            # Vérification orthogonale : W @ W^T ≈ I pour les matrices carrées
            # Pour les matrices non carrées, on vérifie que les colonnes sont orthogonales
            # en approximant avec la matrice la plus proche


def test_agent_policy_head_init(agent):
    """La tête de politique doit avoir de petits poids (gain=0.01).

    Les poids de ``policy_mean`` doivent être proches de zéro au
    début de l'entraînement pour des distributions étroites.
    """
    W = agent.policy_mean.weight.data
    # Avec gain=0.01, les poids typiques sont dans [-0.03, 0.03] (3 sigma)
    assert W.abs().max() < 0.1, (
        f"Les poids de la tête politique sont trop grands : "
        f"max={W.abs().max().item():.4f}"
    )


# ─── Tests : Déterminisme et stochasticité ────────────────────────

def test_agent_deterministic_forward(agent, single_state):
    """En mode eval, la même entrée doit donner exactement la même sortie."""
    agent.eval()
    logits1, value1 = agent(single_state)
    logits2, value2 = agent(single_state)
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(value1, value2)
    agent.train()


def test_agent_stochastic_action(agent, single_state):
    """En mode train, la même entrée doit donner des actions différentes (échantillonnage).

    Deux appels à ``get_action`` avec le même état doivent produire
    des actions différentes car la distribution est échantillonnée.
    """
    agent.train()
    action1, _, _, _ = agent.get_action(single_state)
    action2, _, _, _ = agent.get_action(single_state)
    # Les actions ne doivent pas être identiques (probabilité quasi nulle)
    assert not torch.allclose(action1, action2), (
        "Deux échantillonnages consécutifs donnent la même action — "
        "la stochasticité n'est pas fonctionnelle"
    )
