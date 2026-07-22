"""

Fonction de récompense composite, entièrement paramétrable.

R = R_progress + R_goal + R_heading + R_smooth + R_energy
    + R_stability + R_time - R_collision - R_out - R_flip
    + (bonus agricoles : santé, irrigation, exploration, etc.)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    """Configuration paramétrable de la fonction de récompense composite.

    Regroupe les pondérations et seuils pour les trois familles de termes
    de récompense : navigation, agriculture et tâche d'irrigation (water task).

    Attributes:
        k_progress: Poids de la progression vers l'objectif.
        goal_reward: Récompense de fin d'épisode si l'objectif est atteint.
        goal_radius: Rayon de détection de l'objectif (en mètres).
        heading_weight: Poids de l'erreur de cap.
        smooth_weight: Poids du lissage des actions.
        energy_alpha: Coefficient de pénalité énergétique.
        stability_weight: Poids de la stabilité angulaire.
        time_penalty: Pénalité temporelle constante par pas.
        collision_penalty: Pénalité en cas de collision avec un obstacle.
        out_of_bounds_penalty: Pénalité en cas de sortie des limites.
        flip_penalty: Pénalité en cas de retournement du drone.
        health_bonus: Récompense par unité de pourcentage de maladies traitées.
        irrigation_bonus: Récompense par unité de pourcentage de zones arrosées.
        exploration_bonus: Récompense par unité de pourcentage de cartographie.
        waste_penalty: Pénalité pour gaspillage de pesticide ou d'eau.
        low_battery_penalty: Pénalité si batterie faible sans retour à la base.
        watering_reward: Récompense lors de l'arrosage d'un groupe de plantes.
        refill_reward: Récompense lors du remplissage du réservoir à la bassine.
        refill_threshold: Seuil de niveau du réservoir pour considérer un remplissage.
        time_penalty_per_group: Pénalité temporelle par groupe non arrosé.
        distance_shaping_reward: Récompense pour se rapprocher du groupe le plus proche.
        mission_complete_reward: Bonus de fin de mission quand tous les groupes sont arrosés.
    """
    # Termes de navigation
    k_progress: float = 5.0
    goal_reward: float = 300.0
    goal_radius: float = 0.5
    heading_weight: float = 0.5
    smooth_weight: float = 0.1
    energy_alpha: float = 0.05
    stability_weight: float = 0.3
    time_penalty: float = 0.005
    collision_penalty: float = 50.0
    out_of_bounds_penalty: float = 200.0
    flip_penalty: float = 250.0
    altitude_reward = 0.03
    stability_reward = 0.03
    velocity_reward = 0.02


    # --- Tâche d'irrigation (water task) ---
    watering_reward: float = 50.0           # récompense quand un groupe de plantes est arrosé
    refill_reward: float = 10.0             # récompense quand le réservoir est rempli
    refill_threshold: float = 98.0         # seuil pour considérer un remplissage
    time_penalty_per_group: float = 0.0   # pénalité temporelle par groupe non arrosé
    distance_shaping_reward: float = 5.0  # récompense pour se rapprocher d'un groupe
    mission_complete_reward: float = 500.0  # bonus quand tous les groupes sont arrosés


class RewardCalculator:
    """Calculateur de récompense composite pour l'environnement agricole.

    Fournit trois méthodes de calcul :
        - ``compute`` : récompense de navigation de base.
        - ``compute_agri`` : récompense étendue avec fonctions agricoles.
        - ``compute_water_task`` : récompense dédiée à la tâche d'irrigation
          avec gestion de réservoir d'eau et groupes de plantes.
    """

    def __init__(self, config: RewardConfig):
        """Initialise le calculateur de récompense.

        Args:
            config: Configuration des pondérations et seuils de récompense.
        """
        self.cfg = config
        self._prev_action = np.zeros(4)   # pour le lissage

    def reset(self):
        """Réinitialise la mémoire interne du calculateur de récompense.

        Remet à zéro le vecteur de l'action précédente utilisé pour le
        calcul du terme de lissage.
        """
        self._prev_action = np.zeros(4)

   
        # ------------------------------------------------------------------
    # Méthode pour la tâche d'irrigation avec gestion de ressources
    def compute_water_task(
        self,
        tank_level: float,
        prev_dist: float,
        curr_dist: float,
        drone_pos,
        roll,
        pitch,
        lin_vel,
        just_watered: bool,
        just_refilled: bool,
        all_watered: bool,
        num_unwatered: int,
        crashed: bool
    ) -> tuple[float, dict]:
        """Calcule la récompense globale pour l'étape courante."""
        c = self.cfg
        reward = 0.0
        terms = {}

        # Récompense d'arrosage
        terms["watering"] = c.watering_reward if just_watered else 0.0

        # Récompense de remplissage à la bassine
        terms["refill"] = c.refill_reward if just_refilled else 0.0

        # Pénalité temporelle fixe pour encourager la rapidité
        terms["time_penalty"] = -c.time_penalty
        
        #Distance
        distance_progress = prev_dist - curr_dist
        distance_reward = c.distance_shaping_reward * distance_progress
        reward += distance_reward
        terms["distance"] = distance_reward

        #Altitude
        z = drone_pos[2]
        terms["altitude"] = c.altitude_reward if z > 0.8 else -0.05

        angle = np.sqrt(roll**2 + pitch**2)
        stability = c.stability_reward * np.exp(-angle)
        terms["stability"] = stability

        #Velocity
        speed = np.linalg.norm(lin_vel)
        velocity = c.velocity_reward * np.exp(-0.2*speed)
        terms["velocity"] = velocity
        
        # Bonus de fin de mission
        terms["mission_complete"] = c.mission_complete_reward if all_watered else 0.0

        # Pénalité fatale en cas de crash (sol)
        terms["collision"] = -c.collision_penalty if crashed else 0.0

        total_reward = sum(terms.values())
        return float(total_reward), terms
