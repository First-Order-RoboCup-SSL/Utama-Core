"""Custom BenchMARL task wrapper for SSL scenarios."""

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import pygame
import torch
from benchmarl.environments.common import TaskClass
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from utama_core.rsoccer_simulator.src.Render import (
    COLORS,
    RenderBall,
    RenderSSLRobot,
    SSLRenderField,
)
from utama_core.training.scenario.passing_config import (
    PassingDynamicsConfig,
    PassingRewardConfig,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_scenario import PassingScenario


def _macro_config(n_attackers: int, n_defenders: int) -> PassingScenarioConfig:
    """Create a PassingScenarioConfig with macro-actions enabled."""
    return PassingScenarioConfig(
        n_attackers=n_attackers,
        n_defenders=n_defenders,
        dynamics=PassingDynamicsConfig(use_macro_actions=True, use_unified_actions=False),
        rewards=PassingRewardConfig(
            passer_face_receiver_weight=0.0,  # macros auto-orient
            receiver_face_ball_weight=0.0,  # macros auto-orient
            kick_alignment_weight=0.0,  # KICK_TO auto-aligns
        ),
    )


def _unified_config(n_attackers: int, n_defenders: int) -> PassingScenarioConfig:
    """Create a PassingScenarioConfig with unified action space."""
    return PassingScenarioConfig(
        n_attackers=n_attackers,
        n_defenders=n_defenders,
        dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=True),
        rewards=PassingRewardConfig(
            passer_face_receiver_weight=0.3,  # model controls orientation
            receiver_face_ball_weight=0.3,  # model controls orientation
            kick_alignment_weight=0.0,  # kick gating handles alignment
        ),
    )


# Registry mapping task names to (scenario_class, default_config) pairs
_TASK_REGISTRY: Dict[str, tuple] = {
    # Legacy 6D action space
    "ssl_2v0": (
        PassingScenario,
        PassingScenarioConfig(
            n_attackers=2,
            n_defenders=0,
            dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=False),
        ),
    ),
    "ssl_2v1": (
        PassingScenario,
        PassingScenarioConfig(
            n_attackers=2,
            n_defenders=1,
            dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=False),
        ),
    ),
    "ssl_2v2": (
        PassingScenario,
        PassingScenarioConfig(
            n_attackers=2,
            n_defenders=2,
            dynamics=PassingDynamicsConfig(use_macro_actions=False, use_unified_actions=False),
        ),
    ),
    # Macro-action 3D action space
    "ssl_2v0_macro": (PassingScenario, _macro_config(2, 0)),
    "ssl_2v1_macro": (PassingScenario, _macro_config(2, 1)),
    "ssl_2v2_macro": (PassingScenario, _macro_config(2, 2)),
    # Unified 4D action space [target_x, target_y, target_oren, kick_intent]
    "ssl_2v0_unified": (PassingScenario, _unified_config(2, 0)),
    "ssl_2v1_unified": (PassingScenario, _unified_config(2, 1)),
    "ssl_2v2_unified": (PassingScenario, _unified_config(2, 2)),
}


class SSLTask(TaskClass):
    """BenchMARL TaskClass for SSL training scenarios.

    Usage:
        task = SSLTask.from_name("ssl_2v0")
        # or with custom config:
        task = SSLTask.from_name("ssl_2v0", scenario_config=PassingScenarioConfig(...))
    """

    def __init__(self, name: str, config: dict, scenario_class, scenario_config):
        super().__init__(name=name, config=config)
        self._scenario_class = scenario_class
        self._scenario_config = scenario_config

    @classmethod
    def from_name(
        cls,
        name: str,
        scenario_config: Optional[PassingScenarioConfig] = None,
    ) -> "SSLTask":
        if name not in _TASK_REGISTRY:
            raise ValueError(f"Unknown task '{name}'. Available: {list(_TASK_REGISTRY.keys())}")

        scenario_class, default_config = _TASK_REGISTRY[name]
        if scenario_config is None:
            scenario_config = default_config

        return cls(
            name=name,
            config={"max_steps": scenario_config.max_steps},
            scenario_class=scenario_class,
            scenario_config=scenario_config,
        )

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        scenario = self._scenario_class()
        config = self._scenario_config

        # Each agent gets its own group for independent policies
        group_map: Dict[str, List[str]] = {}
        if config.n_attackers >= 1:
            group_map["passer"] = ["attacker_0"]
        if config.n_attackers >= 2:
            group_map["receiver"] = ["attacker_1"]
        for i in range(config.n_defenders):
            group_map[f"defender_{i}"] = [f"defender_{i}"]

        return lambda: VmasEnv(
            scenario=scenario,
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            max_steps=None,  # Scenario's done() handles termination
            categorical_actions=True,
            clamp_actions=True,
            scenario_config=config,
            group_map=group_map,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self._scenario_config.max_steps

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    # Cached pygame rendering state (shared across calls)
    _render_field: SSLRenderField | None = None
    _render_surface: pygame.Surface | None = None

    @classmethod
    def _ensure_renderer(cls):
        """Lazy-init the pygame surface and field renderer."""
        if cls._render_field is None:
            pygame.init()
            cls._render_field = SSLRenderField()
            cls._render_surface = pygame.Surface(cls._render_field.window_size)

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase) -> torch.Tensor:
        """Render using RSim-style pygame renderer for consistent, stable output."""
        SSLTask._ensure_renderer()
        field_renderer = SSLTask._render_field
        surface = SSLTask._render_surface

        scenario = env._env.scenario

        # Draw field background and markings
        field_renderer.draw(surface)

        def pos_transform(x, y):
            return (
                int(x * field_renderer.scale + field_renderer.center_x),
                int(-y * field_renderer.scale + field_renderer.center_y),
            )

        # Draw attackers (blue team)
        for i, agent in enumerate(scenario.attackers):
            pos = agent.state.pos[0]  # env_index=0
            rot = agent.state.rot[0]
            px, py = pos_transform(pos[0].item(), pos[1].item())
            theta_deg = -math.degrees(rot.item())
            rbt = RenderSSLRobot(px, py, theta_deg, field_renderer.scale, i, COLORS["BLUE"])
            rbt.draw(surface)

        # Draw defenders (yellow team)
        for i, agent in enumerate(scenario.defenders):
            pos = agent.state.pos[0]
            rot = agent.state.rot[0]
            px, py = pos_transform(pos[0].item(), pos[1].item())
            theta_deg = -math.degrees(rot.item())
            rbt = RenderSSLRobot(px, py, theta_deg, field_renderer.scale, i, COLORS["YELLOW"])
            rbt.draw(surface)

        # Draw ball
        bpos = scenario.ball.state.pos[0]
        bx, by = pos_transform(bpos[0].item(), bpos[1].item())
        ball = RenderBall(bx, by, field_renderer.scale)
        ball.draw(surface)

        # Convert pygame surface to numpy RGB array (H, W, 3)
        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)),
            axes=(1, 0, 2),
        )
        return torch.tensor(frame.copy())

    @staticmethod
    def env_name() -> str:
        return "vmas"
