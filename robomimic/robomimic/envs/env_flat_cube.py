import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from robomimic.envs.pymunk_override import DrawOptions
from gym import spaces
import numpy as np
import cv2
import random 


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class TouchCubeEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=128,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # self.rs = np.random.RandomState(seed=2025) # this is bad because we are creating a deterministic setup 


        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        self.reset_state = None # this is the state that we have upon reset 
    
    def generate_spaced_random_points(self, lower, upper, radius = 120, num_points = 5):
        loc_list = list()
        counter = 0 
        while len(loc_list) < num_points:
            counter += 1 
            # candidate_point = rs.randint(lower, upper, size = (2,))
            candidate_point = np.random.randint(lower, upper, size = (2,))
            bad = False
            for point in loc_list:
                if np.linalg.norm(candidate_point - point) < radius:
                    bad = True 
                    break 
            if not bad:
                loc_list.append(candidate_point)
        # print(f"Took {counter} iterations.")
        return loc_list
    def jiggle(self):
        return 2 * (np.random.random() - 0.5)
    
    def equidistant_cube_setup(self):
        cube_1 = [60 + 10 * self.jiggle(), 60 + 10 * self.jiggle()]
        cube_2 = [60 + 10 * self.jiggle(), 450 + 10 * self.jiggle()]
        cube_3 = [450 + 10 * self.jiggle(), 60 + 10 * self.jiggle()]
        cube_4 = [450 + 10 * self.jiggle(), 450 + 10 * self.jiggle()]
        agent = [255 + 10 * self.jiggle(), 255 + 10 * self.jiggle()]
        cube_list = [cube_1, cube_2, cube_3, cube_4]
        random.shuffle(cube_list)
        loc_list = [agent].extend(cube_list)

        rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
        states_list.append(rand_rots)
        state = np.concatenate(states_list)
        return state 

    def reset(self, state = None):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        # state = self.reset_to_state
        if state is None:
            loc_list = self.generate_spaced_random_points(lower = 40, upper = 470, radius = 60, num_points = 5) # placement randomizer for environment 
            # rand_rots = self.rs.randn(4) * 2 * np.pi - np.pi 
            rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 

            # state: [agent, cube 1, cube 2, cube 3, cube 4, rot 1, rot2 , rot3, rot4]
            states_list = loc_list 
            states_list.append(rand_rots)
            state = np.concatenate(states_list)
        # else:
            # print("Resetting to state ", state)
        self._set_state(state)
        self.reset_state = state # useful if we want to replicate the enviornment 

        observation = self._get_obs()
        return observation
    
    def get_state(self):
        return self.reset_state 

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            assert np.all(action < 1.1), "You are using the original 0 -> 512 action space!"
            assert len(np.shape(action)) == 1, "your actions must be flat!"
            rescaled_action = 0.5 * (action + 1) * self.window_size
            # rescaled_action = action * self.window_size # ordinary action is 0 -> 1 
            self.latest_action = rescaled_action
            # print(self.latest_action)
            for i in range(n_steps):
                # Step PD control.
               
                # vel = self.k_p * (rescaled_action - self.agent.position)    # P control works too.
                # self.agent.velocity = vel.tolist()
                # print(rescaled_action, self.agent.position)
                acceleration = self.k_p * (rescaled_action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                acceleration = np.clip(acceleration, -1000, 1000) # limit actions 
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        done = False 
        reward = 0 
        agent_shape = list(self.agent.shapes)[0] # there is only one shape 
        contact_index = -1 
        for i, block in enumerate(self.block_list):
            block_shape = list(block.shapes)[0]
            contact_point = agent_shape.shapes_collide(block_shape).points
            if len(contact_point) > 0: # when you hit something, this should be non-zero 
                done = True 
                reward = 1 
                contact_index = i 
        

        observation = self._get_obs()
        info = self._get_info()
        info["cube_contacted"] = contact_index 

        if np.any(observation["agent_pos"] < -1) or np.any(observation["agent_pos"] > 1):
            done = True 
            reward = -1 
            print("Agent Out of Bounds!")

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        # obs = np.array(
        #     tuple(self.agent.position) \
        #     + tuple(self.block.position) \
        #     + (self.block.angle % (2 * np.pi),))
        state_list = list()
        state_list.append(self.agent.position)
        for block in self.block_list:
            state_list.append(block.position)
        obs = np.concatenate(state_list)
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))

        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            # 'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            # 'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step,
            'cube_contacted' : -1
            }
        return info

    def _render_frame(self, mode, render_size = None):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # canvas.fill((200, 200, 200))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        # goal_body = self._get_goal_pose_body(self.goal_pose)
        # for shape in self.block.shapes:
        #     goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
        #     goal_points += [goal_points[0]]
        #     pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        if render_size is not None:
            img = cv2.resize(img, (render_size, render_size))
        else:
            img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
       
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * self.render_size).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_blocks = state[2:10]
        rot_blocks = state[10:]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.

        for i, block in enumerate(self.block_list):
            block.position = pos_blocks[2 * i : 2 * i + 2]
            block.angle = rot_blocks[i]
        
        # if self.legacy:
        #     # for compatibility with legacy data
        #     self.block.position = pos_block
        #     self.block.angle = rot_block
        # else:
        #     self.block.angle = rot_block
        #     self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 20)

        self.block_list = list()
        self.color_list = ["Blue", "Red", "Green", "Yellow"]
        # self.color_list = [(0, 0, 150), (150, 0, 0), (0, 150, 0), (150, 150, 0)]
        position_list = [(80, 80), (80, 440), (440, 80), (440, 440)]
        for i in range(4):
            self.block_list.append(self.add_cube(position_list[i], height = 40, width = 40, color = self.color_list[i]))
            # self.block_list.append(self.add_cube(position_list[i], height = 20, width = 20, color = self.color_list[i]))

        # for i in range(4):
        #     self.block_list.append(self.add_cube(position_list[i], height = 20, width = 20, color = self.color_list[i]))


        # self.block = self.add_tee((256, 300), 0)
        # self.goal_color = pygame.Color('LightGreen')
        # self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)

        shape.color = pygame.Color('Black')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        # shape.filter = pymunk.ShapeFilter(group=1)

        self.space.add(body, shape)
        return body
    
    def add_cube(self, position, height, width, color):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        if type(color) == str:
            shape.color = pygame.Color(color)
        else:
            shape.color = pygame.Color(*color)
        self.space.add(body, shape)
        return body

    # def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
    #     mass = 1
    #     length = 4
    #     vertices1 = [(-length*scale/2, scale),
    #                              ( length*scale/2, scale),
    #                              ( length*scale/2, 0),
    #                              (-length*scale/2, 0)]
    #     inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
    #     vertices2 = [(-scale/2, scale),
    #                              (-scale/2, length*scale),
    #                              ( scale/2, length*scale),
    #                              ( scale/2, scale)]
    #     inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
    #     body = pymunk.Body(mass, inertia1 + inertia2)
    #     shape1 = pymunk.Poly(body, vertices1)
    #     shape2 = pymunk.Poly(body, vertices2)
    #     shape1.color = pygame.Color(color)
    #     shape2.color = pygame.Color(color)
    #     shape1.filter = pymunk.ShapeFilter(mask=mask)
    #     shape2.filter = pymunk.ShapeFilter(mask=mask)
    #     body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
    #     body.position = position
    #     body.angle = angle
    #     body.friction = 1
    #     self.space.add(body, shape1, shape2)
    #     return body


class TouchCubeImageEnv(TouchCubeEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=128):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            ),
            'states': spaces.Box(
                low=0,
                high=ws,
                shape=(8,),
                dtype=np.float32
            )
        })
        self.render_cache = None
    
    def _get_obs(self, size = None):
        # everything is normalized between 0 -> 1 for ease of training 
        img = super()._render_frame(mode='rgb_array', render_size = size)
        agent_pos = ((np.array(self.agent.position) / self.window_size) - 0.5) * 2
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        state_list = list()

        for block in self.block_list:
            state_list.append(((np.array(block.position) / self.window_size) - 0.5) * 2) # this gives a state from 0 -> 1, well behaved 
        states = np.concatenate(state_list)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos,
            'states' : states 
        }

        # draw action (commmented out)
        # if self.latest_action is not None:
        #     action = np.array(self.latest_action)
        #     coord = (action / 512 * self.render_size).astype(np.int32)
        #     marker_size = int(8/96*self.render_size)
        #     thickness = int(1/96*self.render_size)
        #     cv2.drawMarker(img, coord,
        #         color=(255,0,0), markerType=cv2.MARKER_CROSS,
        #         markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode, height = 90, width = 90):
        # the height and width are just to match the call signature; they do nothing. 
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs(size = height)
        
        return self.render_cache