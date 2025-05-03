from __future__ import annotations

import random
from typing import Any, ClassVar, Tuple, Optional, List, Dict, Any, Union

import numpy as np
import torch

from gymnasium.spaces import Box
# from gymnasium.core import RenderFrame

import pygame

from omnisafe.envs.core import CMDP, env_register
from gym_navigation.envs.navigation_track import NavigationTrack

@env_register
class NavigationTrackSafe(NavigationTrack, CMDP): # MRO matters here
    """
    A 'safe' variant of NavigationTrack that returns an additional cost signal
    if the agent is too close to a wall.

    Omnisafe expects:
    - A .step() method returning (obs, reward, cost, terminated, truncated, info).
    - A .reset() method returning (obs, info).
    - Tensors, not numpy arrays!
    - _support_envs listing custom environment ID.
    """

    # register environment ID, typically in the form of 'env_name-v[0-9]+'
    _support_envs: ClassVar[list[str]] = ['NavigationTrackSafe-v0']

    need_auto_reset_wrapper: bool = True #  automatically resets the environment when an episode ends
    need_time_limit_wrapper: bool = False # no truncation

    _SAFE_DISTANCE = 0.4 # represents 1 meter, the collision threshold is 0.4 meters

    #_num_envs = 1 # number of parallel environments, set to 1 for now

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        track_id: int = 1,
        **kwargs: dict[str, Any]) -> None:
        """OmniSafe will pass env_id and possibly other config in kwargs."""

    
   
        self._screen = None
        # self.render_mode = render_mode
        self._count = 0

        self._num_envs = 1 # number of parallel environments, set to 1 for now

        # Omnisafe expects these properties:
        # - self._observation_space
        # - self._action_space

        NavigationTrack.__init__(self, track_id=track_id)

        self._action_space = Box(
                    low=np.array([0.0, -0.2], dtype=np.float32),
                    high=np.array([0.2, 0.2], dtype=np.float32),
                    dtype=np.float32,
                )
 
        self._observation_space = Box(low=self._SCAN_RANGE_MIN,
                                      high=self._SCAN_RANGE_MAX,
                                      shape=(self._N_OBSERVATIONS,),
                                      dtype=np.float64)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment one timestep.
        Must return (obs, reward, cost, terminated, truncated, info),
        each as a torch.Tensor except info which is a dict.
        """

        # convert incoming action from torch to numpy for the parent and clips it to the allowed action space
        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, self.action_space.low, self.action_space.high)

        obs_np, reward_np, terminated, truncated, info = super().step(action_np)
        cost_value = self._calculate_distance_cost()

        # convert everything to torch tensors for omnisafe
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        reward = torch.as_tensor(reward_np, dtype=torch.float32)
        cost = torch.as_tensor(cost_value, dtype=torch.float32)
        terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool)
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool)

        return obs, reward, cost, terminated_tensor, truncated_tensor, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment for a new episode. Must return (obs, info).
        """
        if seed is not None:
            self.set_seed(seed)

        obs_np, info = super().reset(seed=seed)
        obs = torch.as_tensor(obs_np, dtype=torch.float32)

        self._count = 0
        return obs, info
    
    def _do_perform_action(self, action: np.ndarray) -> None:
        """
        Interpret action as: [linear_speed, angular_speed].
        """
        linear_speed = float(action[0])
        angular_speed = float(action[1])
    
        # If rotating, cap the linear speed to 0.04 or 1/5 of angular speed (like in discrete actions)
        if abs(angular_speed) > 0.0:  
            linear_speed = min(linear_speed, 0.2 * abs(angular_speed))
    
        # Add noise to the action
        linear_speed += self.np_random.normal(0, self._SHIFT_STANDARD_DEVIATION)
        angular_speed += self.np_random.normal(0, self._SHIFT_STANDARD_DEVIATION)

        self._pose.shift(linear_speed, angular_speed)
        self._update_scan()

    def _do_calculate_reward(self, action: np.ndarray) -> float:
        if self._collision_occurred():
            return self._COLLISION_REWARD

        linear_speed = float(action[0])
        angular_speed = float(action[1])

        # reward shaping
        forward_reward = self._FORWARD_REWARD * max(0.0, 5 * linear_speed)
        rotation_penalty = self._ROTATION_REWARD * abs(5 * angular_speed)

        return forward_reward + rotation_penalty

    def _calculate_distance_cost(self) -> float:
        """
        Use sensor readings (self._ranges) from the parent NavigationTrack environment.
        If any sensor reads < 1.0, return cost=1.0; else 0.0.
        """
        if (self._ranges < self._SAFE_DISTANCE).any():
            return 1.0
        return 0.0
    
    def set_seed(self, seed: int) -> None:
        """Set RNG seeds as needed."""
        random.seed(seed)
        np.random.seed(seed)

    def close(self) -> None:
        super().close()

        if self._screen is not None:
            self._screen = None
            pygame.quit()

    
    @property
    def action_space(self):
        return self._action_space


    def render(self) -> np.ndarray:
        """
        Render the environment as an RGB array for OmniSafe's evaluator.
        Returns a numpy array representing the RGB image of the environment.
        """
        # Initialise pygame and screen if needed
        if self._screen is None:
            pygame.init()
            self._screen = pygame.Surface((self._WINDOW_SIZE, self._WINDOW_SIZE))
        
        # Clear screen
        self._screen.fill((255, 255, 255))
        
        # Draw the environment
        self._do_draw(self._screen)
            
        # Visualise the safety constraint boundary
        self._draw_safety_boundary(self._screen)
        
        # Convert surface to numpy array
        array = pygame.surfarray.array3d(self._screen)
        
        # Convert from (width, height, 3) to (height, width, 3)
        array = array.transpose((1, 0, 2))
        return array
    
    def _draw_safety_boundary(self, surface):
        """Draw the safety constraint boundary as a translucent overlay."""
        # Create a transparent surface for the safety boundary
        safety_overlay = pygame.Surface((self._WINDOW_SIZE, self._WINDOW_SIZE), pygame.SRCALPHA)
        
        # Calculate the safety boundary points
        for wall in self._track.walls:
            # Calculate points 1 meter away from each wall segment
            self._draw_dilated_wall(safety_overlay, wall, dilation=self._SAFE_DISTANCE, color=(70, 70, 70, 70)) 
        
        surface.blit(safety_overlay, (0, 0))

    def _draw_dilated_wall(self, surface, wall, dilation, color):
        """Draw a dilated version of a wall to represent safety boundaries."""
        
        # Access x and y coordinates from the Point objects - use x_coordinate and y_coordinate
        start_x, start_y = wall.start.x_coordinate, wall.start.y_coordinate
        end_x, end_y = wall.end.x_coordinate, wall.end.y_coordinate
        
        # Calculate normal vector for the wall
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        
        nx, ny = dy/length, dx/length
        
        # Calculate 4 corners of the dilated wall polygon
        p1 = (start_x + nx * dilation, start_y + ny * dilation)
        p2 = (end_x + nx * dilation, end_y + ny * dilation)
        p3 = (end_x - nx * dilation, end_y - ny * dilation)
        p4 = (start_x - nx * dilation, start_y - ny * dilation)
        
        
        p1_px = (int(p1[0] * self._RESOLUTION) + self._X_OFFSET, 
                self._WINDOW_SIZE - int(p1[1] * self._RESOLUTION) + self._Y_OFFSET)
        p2_px = (int(p2[0] * self._RESOLUTION) + self._X_OFFSET, 
                self._WINDOW_SIZE - int(p2[1] * self._RESOLUTION) + self._Y_OFFSET)
        p3_px = (int(p3[0] * self._RESOLUTION) + self._X_OFFSET, 
                self._WINDOW_SIZE - int(p3[1] * self._RESOLUTION) + self._Y_OFFSET)
        p4_px = (int(p4[0] * self._RESOLUTION) + self._X_OFFSET, 
                self._WINDOW_SIZE - int(p4[1] * self._RESOLUTION) + self._Y_OFFSET)
        
        pygame.draw.polygon(surface, color, [p1_px, p2_px, p3_px, p4_px])
    
    @property
    def action_space(self):
        """Read-only getter for the action space."""
        return self._action_space
 
    @action_space.setter
    def action_space(self, new_space):
        """Allows reassigning action_space if needed."""
        self._action_space = new_space
 
    @property
    def observation_space(self):
        return self._observation_space
 
    @observation_space.setter
    def observation_space(self, new_space):
        self._observation_space = new_space



