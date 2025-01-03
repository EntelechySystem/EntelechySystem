import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
# from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from .core import Agent

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"  # #HACK 这个似乎无用


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
            self,
            scenario,
            world,
            max_cycles,
            render_mode=None,
            continuous_actions=False,
            local_ratio=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c
            if agent.具有视觉:
                视觉_dim = self.world.dim_视觉
            if agent.具有听觉:
                听觉_dim = self.world.dim_听觉
            if agent.具有表情:
                表情_dim = self.world.dim_表情
            if agent.具有说话能力:
                说话_dim = self.world.dim_说话
            if agent.具有抓取运动能力:
                抓取运动_dim = self.world.dim_抓取运动
            if agent.需要睡眠:
                睡眠_dim = self.world.dim_睡眠
            if agent.需要饮食:
                饮食_dim = self.world.dim_饮食

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:  # #HACK 这个 continuous_actions 似乎没有用了
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Dict({
                    '移动运动': spaces.Discrete(space_dim),
                    '说话': spaces.MultiDiscrete([0x10FFFF + 1] * 说话_dim),
                    # '说话': spaces.Box(low=0, high=0x10FFFF, shape=(256,), dtype=np.uint32),  # 备选方案
                    '表情': spaces.Discrete(表情_dim),
                    '抓取运动': spaces.Discrete(抓取运动_dim),
                    '睡眠': spaces.Discrete(睡眠_dim),
                    '饮食': spaces.Discrete(饮食_dim),
                })
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    # def observation(self, agent):
    #     visible_entities = self.world.get_visible_entities(agent)
    #     entity_pos = []
    #     for entity in visible_entities:
    #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     return np.concatenate([agent.state.p_vel] + entity_pos)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action['移动运动'] % mdim)
                    action['移动运动'] //= mdim

            # if not agent.silent:
            #     cdim = self.world.dim_c
            #     scenario_action.append(action[0:cdim])
            #     action = action[cdim:]

            # if agent.具有视觉:
            #     scenario_action.append(action[0:self.world.dim_视觉])
            #     action = action[self.world.dim_视觉:]
            # if agent.具有听觉:
            #     scenario_action.append(action[0:self.world.dim_听觉])
            #     action = action[self.world.dim_听觉:]
            # if agent.具有说话能力:
            #     scenario_action.append(action[0:self.world.dim_说话])
            #     action = action[self.world.dim_说话:]
            # if agent.具有触觉:
            #     scenario_action.append(action[0:self.world.dim_触觉])
            #     action = action[self.world.dim_触觉:]
            # if agent.具有嗅觉:
            #     scenario_action.append(action[0:self.world.dim_嗅觉])
            #     action = action[self.world.dim_嗅觉:]
            # if agent.具有温度知觉:
            #     scenario_action.append(action[0:self.world.dim_温度知觉])
            #     action = action[self.world.dim_温度知觉:]
            # if agent.具有疼痛知觉:
            #     scenario_action.append(action[0:self.world.dim_疼痛知觉])
            if agent.具有说话能力:
                scenario_action.append(action['说话'])
            if agent.具有表情:
                scenario_action.append(action['表情'])
            if agent.具有抓取运动能力:
                scenario_action.append(action['抓取运动'])
            if agent.需要睡眠:
                scenario_action.append(action['睡眠'])
            if agent.需要饮食:
                scenario_action.append(action['饮食'])

            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                        global_reward * (1 - self.local_ratio)
                        + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        agent.action.说话 = np.zeros(self.world.dim_说话)
        agent.action.表情 = np.zeros(self.world.dim_表情)
        agent.action.抓取运动 = np.zeros(self.world.dim_抓取运动)
        agent.action.睡眠 = np.zeros(self.world.dim_睡眠)
        agent.action.饮食 = np.zeros(self.world.dim_饮食)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        # if not agent.silent:
        #     # communication action
        #     if self.continuous_actions:
        #         agent.action.c = action[0]
        #     else:
        #         agent.action.c = np.zeros(self.world.dim_c)
        #         agent.action.c[action[0]] = 1.0
        if agent.具有说话能力:
            # agent.action.说话 = action[0: self.world.dim_说话]
            # action = action[self.world.dim_说话:]
            agent.action.说话 = action[0]
            action = action[1:]
        if agent.具有表情:
            agent.action.表情 = action[0]
            action = action[1:]
        if agent.具有抓取运动能力:
            agent.action.抓取运动 = action[0]
            action = action[1:]
        if agent.需要睡眠:
            agent.action.睡眠 = action[0]
            action = action[1:]
        if agent.需要饮食:
            agent.action.饮食 = action[0]
            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                    (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                    0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                            "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
