You need to fundamentally update the MDP observation space, physical step logic, and reward function to introduce finite resource management and spatial task planning. Please update `environment/agri_drone_env.py` and `environment/reward/reward_function.py` based on the following strict rules.

**1. Observation Space Updates (The State)**
Expand the `gym.spaces.Box` observation space to include the following new variables, properly flattened and normalized:

- **Water Basin Coordinates:** Add the `(x, y, z)` spatial coordinates of a static water basin.
- **Water Tank Level:** Add a continuous variable for the drone's internal water tank capacity. It initializes at 100.0 (100%) at the start of the episode.
- **Plant Groups Matrix:** Add a structured array (flattened) representing the plant groups that need watering. Each group is defined by 4 features: `[x, y, z, is_watered]`. The `is_watered` state is a flag (0.0 for False, 1.0 for True).

**2. Environment Mechanics (Step Logic)**

- **Water Consumption:** When the drone is above an unwatered plant group and successfully activates its watering action, the `is_watered` state of that specific group updates to 1.0, and the drone's tank level decreases by exactly 2.0 (2%).
- **Basin Refilling:** If the drone flies to the water basin coordinates (within a defined proximity radius), its tank completely refills to 100.0.

**3. Balanced Reward Function Overhaul**
Update the reward calculator to push the agent towards exploration without causing premature suicide policies:

- **Watering Action:** Award a sparse +5.0 reward immediately when a plant group transitions from unwatered to watered.
- **Refill Action:** Award a +1.0 reward when the drone refills its tank at the basin (only trigger this if the tank was below 98.0 to prevent infinite reward farming).
- **Scaled Time Penalty:** Apply a small scaled penalty for time elapsed. At every step, apply `-0.02 * number_of_unwatered_groups`. This provides urgency without overwhelmingly negative returns.
- **Distance Reward Shaping (Crucial):** At every step, calculate the distance to the nearest _unwatered_ plant group. If the drone moved closer to it compared to the previous step, award a small positive shaping reward (e.g., +0.05) to guide the exploration and offset the time penalty.
- **Mission Complete (Early Stopping):** Check the plant matrix at every step. If ALL plant groups have `is_watered == 1.0`, immediately terminate the episode (`terminated = True`) and award a massive +100.0 completion bonus.

Please adjust the `observation_space` dimensions in `__init__`, implement the state matrix flattening in `_get_obs()`, and meticulously rewrite the reward logic.
