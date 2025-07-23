import functools
import elements
import embodied
import gym
import numpy as np
from .polyomino_env import PolyominoEnvironment


class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      if env == 'PolyominoEnv':
        print("Using PolyominoEnvironment")
        self._env = PolyominoEnvironment()
        self._polyomino_env = True
      else:
        self._env = gym.make(env, **kwargs)
        self._polyomino_env = False
    else:
      assert not kwargs, kwargs
      self._env = env
      self._polyomino_env = False
    
    # Special handling for PolyominoEnv's dict observation
    if self._polyomino_env:
      self._obs_dict = True  # It has dict observations but we'll handle specially
    else:
      self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._polyomino_env:
      # For PolyominoEnv, create a single concatenated image space
      spaces = {self._obs_key: gym.spaces.Box(low=0, high=255, shape=(128, 256, 1), dtype=np.uint8)}
    elif self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      if self._polyomino_env:
        obs = self._env.reset()
        # Check if reset returns tuple (new gym API) or just obs (old API)
        if isinstance(obs, tuple):
          obs, info = obs
        obs = self._process_polyomino_obs(obs)
      else:
        obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    
    if self._polyomino_env:
      step_result = self._env.step(action)
      if len(step_result) == 5:  # New gym API: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, self._info = step_result
        self._done = terminated or truncated
      else:  # Old gym API: obs, reward, done, info
        obs, reward, self._done, self._info = step_result
      obs = self._process_polyomino_obs(obs)
    else:
      obs, reward, self._done, self._info = self._env.step(action)
    
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if self._polyomino_env:
      # obs is already processed into a single image
      obs = {self._obs_key: obs}
    elif not self._obs_dict:
      obs = {self._obs_key: obs}
    else:
      obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    if self._polyomino_env:
      # For PolyominoEnv, we need to handle the dict observation
      # Get the current observation and convert it to RGB for rendering
      try:
        # Try to get current state if available
        if hasattr(self._env, 'latest_env_state') and self._env.latest_env_state:
          obs_dict = self._env.latest_env_state.get('observation', {})
          if 'left' in obs_dict and 'right' in obs_dict:
            combined = self._process_polyomino_obs(obs_dict)
            # Convert grayscale to RGB for rendering
            return np.repeat(combined, 3, axis=2)
      except:
        pass
      # Fallback: return a placeholder image
      return np.zeros((128, 256, 3), dtype=np.uint8)
    else:
      image = self._env.render('rgb_array')
      assert image is not None
      return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)

  def _process_polyomino_obs(self, obs_dict):
    """Convert PolyominoEnv dict observation to single image."""
    left = obs_dict['left']   # (128, 128, 1)
    right = obs_dict['right'] # (128, 128, 1)
    # Concatenate horizontally to create (128, 256, 1)
    combined = np.concatenate([left, right], axis=1)
    return combined
