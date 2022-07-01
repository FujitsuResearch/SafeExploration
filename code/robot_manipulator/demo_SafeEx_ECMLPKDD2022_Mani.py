#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gym
from gym import spaces
from gym.utils import seeding
from scipy.stats import norm


import pandas as pd



# %%
# Simiulation parameters and Hyperparameters

max_steps=100       # Number of time steps in one episode (default: 100)

max_episodes=100    # Number of episodes to train (default: 100)
logger_interval=50  # Logger_interval (default: 50)
num_trials=10       # Number of trials (default: 10)

gamma=0.99          # Discount rate (default: 0.99)
lr_actor=1e-3       # Learning rate of actor network (default: 1e-3)
lr_critic=2e-3      # Learning rate of critic network (default: 2e-3)


mu_q1=0.0           # Mean of disturbances in angle of link 1  (default: 0)
sigma_q1=0.01       # Standard deviation of disturbances in angle of link 1  (default: 0.01)
mu_q2=0.1           # Mean of disturbances in angle of link 2  (default: 0.1)
sigma_q2=0.03       # Standard deviation of disturbances in angle of link 2  (default: 0.03)
mu_q1dot=-0.1       # Mean of disturbances in rotation speed of link 1  (default: -0.1)
sigma_q1dot=0.02    # Standard deviation of disturbances in rotation speed of link 1 (default: 0.02)
mu_q2dot=0.05       # Mean of disturbances in rotation speed of link 2  (default: 0.05)
sigma_q2dot=0.01    # Standard deviation of disturbances in rotation speed of link 2 (default: 0.01)

seed=0              # Random seed (exploration) (default: 0, random: -1)


load_path='./result_paper/'      # Path to load results of paper (default: .'./result_paper/')
save_path='./'      # Path to save models and results (default: ./)
header='demo_'      # Header of saved models and results of paper (default: 'demo_')



# %%
# DDPG algorithms and models
# Code of DDPG algorithm is given in https://keras.io/examples/rl/ddpg_pendulum/

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt*0
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)*0
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


# %%
class Buffer:
    def __init__(self,num_states, num_actions, target_actor, target_critic, actor_model, critic_model, 
                 actor_optimizer, critic_optimizer,
                 buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        
        self.target_actor =target_actor
        self.target_critic =target_critic
        self.actor_model = actor_model
        self.critic_model = critic_model
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

    


# %%
def get_actor(num_states,num_actions):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * 2
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states,num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# %%
def policy(state, noise_object,actor_model):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise*0

    # We make sure action is within bounds
    legal_action = sampled_actions
    
    return [np.squeeze(legal_action)]


# %%
# Envirionment and proposed method

class Mani_ECMLPKDD2022(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, mu_q1=0, sigma_q1=0.05, mu_q2=0.5, sigma_q2=0.05,
                         mu_q1dot=0, sigma_q1dot=0.05, mu_q2dot=0.5, sigma_q2dot=0.05):
        self.max_speed = 10
        self.const_speed = 6 
        self.max_torque =2.
        self.dt = .05
        self.m11_hat = 3.91*1e-3
        self.m22_hat = 2.39*1e-3
        self.d11_hat = 9.37*1e-3
        self.d22_hat = 9.37*1e-3
        self.V1 = 9.0108*1e-2
        self.V2 = 1.9183*1e-2
        self.alpha = 6.89*1e-2
        
        self.viewer = None
       
        self.mu_q1=mu_q1
        self.sigma_q1=sigma_q1
        self.mu_q2=mu_q2
        self.sigma_q2=sigma_q2
        self.mu_q1dot=mu_q1dot
        self.sigma_q1dot=sigma_q1dot
        self.mu_q2dot=mu_q2dot
        self.sigma_q2dot=sigma_q2dot
        
        self.n_c=4
        self.eta=0.95
        self.tau=2
        self.xi=0.9998
        self.xi_dash=1-(1-self.xi)/self.n_c


        high = np.array([1., 1., 1., 1.,  self.max_speed, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.max_torque, -self.max_torque], dtype=np.float32),
            high=np.array([self.max_torque, self.max_torque], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u,step):
        q1,q2, q1dot,q2dot= self.state  # th := theta

        m11_hat = self.m11_hat
        m22_hat = self.m22_hat
        d11_hat = self.d11_hat
        d22_hat = self.d22_hat
        V1 = self.V1
        V2 = self.V2
        alpha = self.alpha
        
        dt = self.dt
        u = u[0]
        
                

        costs = 2*(angle_normalize(q1)) ** 2 + 2*(angle_normalize(q2-5*np.pi/6)) ** 2 \
                + 0.1 * q1dot ** 2 + 0.1 * q2dot ** 2 + 0.001 * (u[0] ** 2) + 0.001 * (u[1] ** 2)
        
    
        w_q1    = np.random.normal(self.mu_q1,self.sigma_q1,1)[0]
        w_q1dot = np.random.normal(self.mu_q1dot,self.sigma_q1dot,1)[0]
        w_q2    = np.random.normal(self.mu_q2,self.sigma_q2,1)[0]
        w_q2dot = np.random.normal(self.mu_q2dot,self.sigma_q2dot,1)[0]
        
        new_q1dot = q1dot - dt*(d11_hat/m11_hat)*q1dot -dt*(V1/m11_hat)*np.cos(q1) + dt*(alpha/m11_hat)*u[0] + w_q1dot
        new_q2dot = q2dot - dt*(d22_hat/m22_hat)*q2dot -dt*(V2/m22_hat)*np.cos(q2) + dt*(alpha/m22_hat)*u[1] + w_q2dot
        new_q1 = q1 +  dt * q1dot +  w_q1 
        new_q2 = q2 +  dt * q2dot +  w_q2 
        new_q1dot = np.clip(new_q1dot, -self.max_speed, self.max_speed)
        new_q2dot = np.clip(new_q2dot, -self.max_speed, self.max_speed)
        
        if (np.abs(new_q1dot) > self.const_speed) or (np.abs(new_q2dot) > self.const_speed): 
            const_violation = 1
        else:
            const_violation = 0
                         

        self.state = np.array([new_q1,new_q2,new_q1dot,new_q2dot])
        return self._get_obs(), -costs, False, {}, self.state[0:2], const_violation

    def reset(self,ini_state=False):
        if ini_state is False:
            high = np.array([np.pi,np.pi, 1, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:       
            self.state=ini_state
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        q1,q2, q1dot,q2dot = self.state
        return np.array([np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2), q1dot,q2dot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def auto_u_adj(self, u, step, ex):
        q1,q2, q1dot,q2dot = self.state  # th := theta

        m11_hat = self.m11_hat
        m22_hat = self.m22_hat
        d11_hat = self.d11_hat
        d22_hat = self.d22_hat
        V1 = self.V1
        V2 = self.V2
        alpha = self.alpha
        
        dt = self.dt
        u = u[0]
        
        a1 = dt*(d11_hat/m11_hat)
        a2 = dt*(d22_hat/m22_hat)
        b1=  dt * (alpha/m11_hat) 
        b2=  dt * (alpha/m22_hat)
        
        delta12_bar = dt*(V1/m11_hat) 
        delta34_bar = dt*(V2/m22_hat) 
        delta12_hat_bar = np.abs(2-dt*(d11_hat/m11_hat))*dt*(V1/m11_hat) 
        delta34_hat_bar = np.abs(2-dt*(d22_hat/m22_hat))*dt*(V2/m22_hat) 
        
        eta=self.eta
        tau=self.tau
        n_c=self.n_c
        xi=self.xi
        xi_dash=self.xi_dash
        
        eta_dash_k=1-(1-(eta/(xi**(step)))**(1/tau))/n_c
        phi_etadash_k = norm.ppf(eta_dash_k, 0)
        
        phi_xidash = norm.ppf(xi_dash, 0)
        
        mu_q1dot=self.mu_q1dot
        mu_q2dot=self.mu_q2dot
        sigma_q1dot=self.sigma_q1dot
        sigma_q2dot=self.sigma_q2dot
        
        q1dot_max=self.const_speed
        q2dot_max=self.const_speed
        q1dot_min=-self.const_speed
        q2dot_min=-self.const_speed
        
        
        if (np.abs(q1dot)<=q1dot_max) and (np.abs(q2dot)<=q2dot_max):
            if ((sigma_q1dot <= (1/phi_etadash_k) * (q1dot_max - (1-a1)*q1dot - b1*u[0] - mu_q1dot -delta12_bar ) ) 
            and (sigma_q1dot <= (1/phi_etadash_k) * (-q1dot_min + (1-a1)*q1dot + b1*u[0] + mu_q1dot -delta12_bar ) ) 
            and (sigma_q2dot <= (1/phi_etadash_k) * (q2dot_max - (1-a2)*q2dot - b2*u[1] - mu_q2dot -delta34_bar ) ) 
            and (sigma_q2dot <= (1/phi_etadash_k) * (-q2dot_min + (1-a2)*q2dot + b2*u[1] + mu_q2dot -delta34_bar ) )) :
                sigma_k1 = (1/ b1) * np.sqrt(  ( ((1/phi_etadash_k) * (q1dot_max - (1-a1)*q1dot - b1*u[0] - mu_q1dot -delta12_bar  ))**2 
                                                - sigma_q1dot**2))
                sigma_k2 = (1/ b1) * np.sqrt(  ( ((1/phi_etadash_k) * (-q1dot_min + (1-a1)*q1dot + b1*u[0] + mu_q1dot -delta12_bar  ))**2 
                                                - sigma_q1dot**2))
                sigma_k3 = (1/ b2) * np.sqrt(  ( ((1/phi_etadash_k) * (q2dot_max - (1-a2)*q2dot - b2*u[1] - mu_q2dot -delta34_bar  ))**2 
                                                - sigma_q2dot**2))
                sigma_k4 = (1/ b2) * np.sqrt(  ( ((1/phi_etadash_k) * (-q2dot_min + (1-a2)*q2dot + b2*u[1] + mu_q2dot -delta34_bar  ))**2 
                                                - sigma_q2dot**2))
                sigma_k=np.min([sigma_k1,sigma_k2,sigma_k3,sigma_k4])
                if ex:
                    u = u + np.random.normal(0,sigma_k,2)                    
                u1=u[0]
                u2=u[1]
                u=[u1,u2]
                u_type = "ex"
            else:
                u1_stay_ub = (1/b1) * ( q1dot_max - (1-a1)*q1dot - (1-a1)*mu_q1dot - delta12_bar - phi_etadash_k*sigma_q1dot)
                u1_stay_lb = (1/b1) * ( q1dot_min - (1-a1)*q1dot - (1-a1)*mu_q1dot + delta12_bar + phi_etadash_k*sigma_q1dot)
                u2_stay_ub = (1/b2) * ( q2dot_max - (1-a2)*q2dot - (1-a2)*mu_q2dot - delta34_bar - phi_etadash_k*sigma_q2dot)
                u2_stay_lb = (1/b2) * ( q2dot_min - (1-a2)*q2dot - (1-a2)*mu_q2dot + delta34_bar + phi_etadash_k*sigma_q2dot)
                u1 = - ((1-a1)*q1dot+(1-a1)*mu_q1dot)/b1
                u2 = - ((1-a2)*q1dot+(1-a2)*mu_q2dot)/b2
                if u1 > u1_stay_ub:
                    u1=u1_stay_ub
                elif u1< u1_stay_lb:
                    u1=u1_stay_lb                    
                if u2 > u2_stay_ub:
                    u2=u2_stay_ub
                elif u2< u2_stay_lb:
                    u2=u2_stay_lb  
                u=[u1,u2]
                u_type = "stay"
        else:
            u1_back_ub = (1/b1) * ( q1dot_max - ((1-a1)**2)*q1dot -(2-a1)*mu_q1dot - delta12_hat_bar 
                                  - phi_xidash*np.sqrt((1-a1)**2 +1)*sigma_q1dot)
            u1_back_lb = (1/b1) * ( q1dot_min - ((1-a1)**2)*q1dot -(2-a1)*mu_q1dot + delta12_hat_bar 
                                   + phi_xidash*np.sqrt((1-a1)**2 +1)*sigma_q1dot)
            u2_back_ub = (1/b2) * ( q2dot_max - ((1-a2)**2)*q1dot -(2-a2)*mu_q2dot - delta34_hat_bar 
                                  - phi_xidash*np.sqrt((1-a2)**2 +1)*sigma_q2dot)
            u2_back_lb = (1/b2) * ( q2dot_min - ((1-a2)**2)*q1dot -(2-a2)*mu_q2dot + delta34_hat_bar 
                                   + phi_xidash*np.sqrt((1-a2)**2 +1)*sigma_q2dot)
            u1 = - (((1-a1)**2)*q1dot + (2-a1)*mu_q1dot)/((1-a1)*b1)          
            u2 = - (((1-a2)**2)*q2dot + (2-a2)*mu_q2dot)/((1-a2)*b2)          
            if u1 > u1_back_ub:
                u1=u1_back_ub
            elif u1< u1_back_lb:
                u1=u1_back_lb
            if u2 > u2_back_ub:
                u2=u2_back_ub
            elif u2< u2_back_lb:
                u2=u2_back_lb
            u=[u1,u2]
            u_type = "back"
            
        return [u], u_type    

    def auto_u_adj_IFAC(self, u, step, ex):
        q1,q2, q1dot,q2dot = self.state  # th := theta

        m11_hat = self.m11_hat
        m22_hat = self.m22_hat
        d11_hat = self.d11_hat
        d22_hat = self.d22_hat
        V1 = self.V1
        V2 = self.V2
        alpha = self.alpha
        
        dt = self.dt
        u = u[0]
        
        a1 = dt*(d11_hat/m11_hat)
        a2 = dt*(d22_hat/m22_hat)
        b1=  dt * (alpha/m11_hat) 
        b2=  dt * (alpha/m22_hat)
        
        delta12_bar = dt*(V1/m11_hat) 
        delta34_bar = dt*(V2/m22_hat) 
        delta12_hat_bar = np.abs(2-dt*(d11_hat/m11_hat))*dt*(V1/m11_hat) 
        delta34_hat_bar = np.abs(2-dt*(d22_hat/m22_hat))*dt*(V2/m22_hat) 
        
        eta=self.eta
        tau=self.tau
        n_c=self.n_c
        
        eta_dash_0=1-(1-(eta)**(1/tau))/n_c
        phi_etadash_0 = norm.ppf(eta_dash_0, 0)                
       
        mu_q1dot=self.mu_q1dot
        mu_q2dot=self.mu_q2dot
        sigma_q1dot=self.sigma_q1dot
        sigma_q2dot=self.sigma_q2dot
        
        q1dot_max=self.const_speed
        q2dot_max=self.const_speed
        q1dot_min=-self.const_speed
        q2dot_min=-self.const_speed
        
        if (np.abs(q1dot)<=q1dot_max) and (np.abs(q2dot)<=q2dot_max):
            if ((sigma_q1dot <= (1/phi_etadash_0) * (q1dot_max - (1-a1)*q1dot - b1*u[0]  -delta12_bar ) ) 
            and (sigma_q1dot <= (1/phi_etadash_0) * (-q1dot_min + (1-a1)*q1dot + b1*u[0]  -delta12_bar ) ) 
            and (sigma_q2dot <= (1/phi_etadash_0) * (q2dot_max - (1-a2)*q2dot - b2*u[1]  -delta34_bar ) ) 
            and (sigma_q2dot <= (1/phi_etadash_0) * (-q2dot_min + (1-a2)*q2dot + b2*u[1]  -delta34_bar ) )) : 
                sigma_k1 = (1/ b1) * np.sqrt(  ( ((1/phi_etadash_0) * (q1dot_max - (1-a1)*q1dot - b1*u[0]  -delta12_bar  ))**2 
                                                - sigma_q1dot**2))
                sigma_k2 = (1/ b1) * np.sqrt(  ( ((1/phi_etadash_0) * (-q1dot_min + (1-a1)*q1dot + b1*u[0]  -delta12_bar  ))**2 
                                                - sigma_q1dot**2))
                sigma_k3 = (1/ b2) * np.sqrt(  ( ((1/phi_etadash_0) * (q2dot_max - (1-a2)*q2dot - b2*u[1]  -delta34_bar  ))**2 
                                                - sigma_q2dot**2))
                sigma_k4 = (1/ b2) * np.sqrt(  ( ((1/phi_etadash_0) * (-q2dot_min + (1-a2)*q2dot + b2*u[1]  -delta34_bar  ))**2 
                                                - sigma_q2dot**2))
                sigma_k=np.min([sigma_k1,sigma_k2,sigma_k3,sigma_k4])
                if ex:
                    u = u + np.random.normal(0,sigma_k,2)
                u1=u[0]
                u2=u[1]
                u=[u1,u2]
                
                u_type = "ex"
            else:
                u1 = - ((1-a1)*q1dot)/b1*0
                u2 = - ((1-a2)*q1dot)/b2*0
                u=[u1,u2]
                u_type = "stay"
        else:
            u1 = - (((1-a1)**2)*q1dot )/((1-a1)*b1)          
            u2 = - (((1-a2)**2)*q2dot )/((1-a2)*b2)          
            u=[u1,u2]
            u_type = "back"
            
        return [u], u_type    

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)






# %%


def main():
    
    ####  hyperparameters of DDPG algorithm  ####
    
    memory_capacity = 5e5 # 1e6 
    tau_DDPG = 5e-3 #1e-3  
    epsilon = 1.0  
    batch_size = 64
    weight_decay = 0# 1e-2
    
    ##############################################


    method_list = ["OKAWAetal2020", "PROPOSED" ]
    
    res_state_all=[]
    res_q1dot_all=[]
    res_q2dot_all=[]
    res_q1_all=[]
    res_q2_all=[]
    res_u_all=[]
    res_u1_all=[]
    res_u2_all=[]

    for method_ID in [0,1]: 

        method=method_list[method_ID]
        print("Method: " + method)
    
        ActorNet_name= load_path +header + "ActorNet_" + method  +"_Mani.h5"
        CriticNet_name= load_path + header + "CriticNet_" + method +"_Mani.h5"     

        env = Mani_ECMLPKDD2022(mu_q1, sigma_q1, mu_q2, sigma_q2, 
                      mu_q1dot, sigma_q1dot, mu_q2dot, sigma_q2dot)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        no_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0) * np.ones(1))
        
        actor_model = get_actor(num_states,num_actions)
        critic_model = get_critic(num_states,num_actions)
        actor_model.load_weights(ActorNet_name)
        critic_model.load_weights(CriticNet_name)


        res_state_all.append([])
        res_q1dot_all.append([])
        res_q2dot_all.append([])
        res_q1_all.append([])
        res_q2_all.append([])
        res_u1_all.append([])
        res_u2_all.append([])


        ini_state = np.array([np.pi, np.pi, 0, 0])
        for episode in range(1):
            prev_state = env.reset(ini_state=ini_state)
            res_q1dot_all[method_ID].append(prev_state[4])
            res_q2dot_all[method_ID].append(prev_state[5])
            res_q1_all[method_ID].append(ini_state[0])
            res_q2_all[method_ID].append(ini_state[1])
            for step in range(max_steps):
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = policy(tf_prev_state, no_noise,actor_model)
                if method == method_list[0]: 
                    action, u_type = env.auto_u_adj_IFAC(action, step, False)
                elif method ==method_list[1]:
                    action, u_type = env.auto_u_adj(action, step, False)
                state, reward_gd, done, info, q1q2 ,const_x = env.step(action, step)
                prev_state = state

                res_q1dot_all[method_ID].append(prev_state[4])
                res_q2dot_all[method_ID].append(prev_state[5])
                q1= angle_normalize(q1q2[0])
                q2= angle_normalize(q1q2[1])
                res_q1_all[method_ID].append(q1)
                res_q2_all[method_ID].append(q2)
                res_u1_all[method_ID].append(action[0][0])
                res_u2_all[method_ID].append(action[0][1])

                if done:
                    break
                    
            res_u1_all[method_ID].append([])
            res_u2_all[method_ID].append([])

        env.close()
        
        df_res_eval = pd.DataFrame(columns=['q1', 'q2', 'q1dot', 'q2dot', 'u1', 'u2'])  
        for step in range( max_steps+1):
            df_res_eval = df_res_eval.append({'q1': res_q1_all[method_ID][step], 
                                        'q2': res_q2_all[method_ID][step], 
                                        'q1dot': res_q1dot_all[method_ID][step], 
                                        'q2dot': res_q2dot_all[method_ID][step], 
                                        'u1': res_u1_all[method_ID][step], 
                                        'u2': res_u2_all[method_ID][step]}, ignore_index=True)
            
        df_res_eval.to_csv(save_path + header +'res_eval_' +method + '_Mani.csv')
        df_res_reward=pd.DataFrame(res_q1_all[method_ID])

if __name__ == '__main__':
    
    main()

