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

max_steps=100        # 'Number of time steps in one episode (default: 100)'
max_episodes=100     # 'Number of episodes to train (default: 100)'
logger_interval=50   # 'Logger_interval (default: 50)'
num_trials=10        # 'Number of trials (default: 10)'

gamma=0.99           # 'Discount rate (default: 0.99)'
lr_actor=1e-3        # 'Learning rate of actor network (default: 1e-3)'
lr_critic=2e-3       # 'Learning rate of critic network (default: 2e-3)'


mu_th=0.0            # 'Mean of disturbances in angle  (default: 0)'
sigma_th=0.05        # 'Standard deviation of disturbances in angle  (default: 0.05)'
mu_thdot=0.5         # 'Mean of disturbances in angler velocity  (default: 0.5)'
sigma_thdot=0.1      # 'Standard deviation of disturbances in angler velocity (default: 0.1)'

seed=0               # 'Random seed (exploration) (default: 0, random: -1)'

load_path='./'       # 'Path to load results (default: ./)'
save_path='./'       # Path to save models and results (default: ./)
header=''            # 'Header of saved models and results of paper (default: empty)'



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
def get_actor(num_states):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 5.0 for Pendulum.
    outputs = outputs * 5
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
    outputs = layers.Dense(1)(out)

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
#     legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    legal_action = sampled_actions
    
    return [np.squeeze(legal_action)]


# %%
# Envirionment and proposed method

class Pend_ECMLPKDD2022(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, mu_th=0, sigma_th=0.05, mu_thdot=.5, sigma_thdot=0.1):
        self.max_speed = 8
        self.max_torque =5.
        
        self.const_speed = 6 
        
        self.Ts = .05
        self.g = 9.8
        self.m = 1.
        self.l = 1.
        self.viewer = None
        
        self.mu_th=mu_th
        self.sigma_th=sigma_th
        self.mu_thdot=mu_thdot
        self.sigma_thdot=sigma_thdot
        
        self.n_c=2
        self.eta=0.95
        self.tau=2
        self.xi=0.9998
        
        self.xi_dash=1-(1-self.xi)/self.n_c


        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
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

    def step(self, u):
        th, thdot = self.state  

        g = self.g
        m = self.m
        l = self.l
        Ts = self.Ts
        u = u[0]

        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    
        w_th    = np.random.normal(self.mu_th,self.sigma_th,1)[0]
        w_thdot = np.random.normal(self.mu_thdot,self.sigma_thdot,1)[0]
        
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * Ts +w_thdot
        newth = th + thdot * Ts +  w_th 
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        
        if np.abs(newthdot) > self.const_speed:
            const_violation = 1
        else:
            const_violation = 0
                         

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}, self.state[0], const_violation

    def reset(self,ini_state=False):
        if ini_state is False:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:       
            self.state=ini_state
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

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
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        Ts = self.Ts
        u = u[0]
        B2= 3. *Ts/(m*l**2)
        delta_bar = 3. * g*Ts/ (2*l)
        Delta_bar = 2 * 3. * g*Ts/ (2*l)
        
        eta=self.eta
        tau=self.tau
        n_c=self.n_c
        xi=self.xi
        xi_dash=self.xi_dash
        
        eta_dash_k=1-(1-(eta/(xi**(step)))**(1/tau))/n_c
        phi_etadash_k = norm.ppf(eta_dash_k, 0)
        
        phi_xidash = norm.ppf(xi_dash, 0)
        
        mu_thdot=self.mu_thdot
        sigma_thdot=self.sigma_thdot
        
        thdot_max=self.const_speed
        thdot_min=-self.const_speed
        
        
        if (thdot<=thdot_max) and (thdot>=thdot_min):
            if (sigma_thdot <= (1/phi_etadash_k) * (thdot_max - thdot - B2*u - mu_thdot -delta_bar ) ) \
            and (sigma_thdot <= (1/phi_etadash_k) * (-thdot_min + thdot + B2*u + mu_thdot - delta_bar ) ): 
                sigma_k1 = (1/ B2) * np.sqrt(  ( ((1/phi_etadash_k) * (thdot_max - thdot - B2*u - mu_thdot -delta_bar ))**2 - sigma_thdot**2))
                sigma_k2 = (1/ B2) * np.sqrt(  ( ((1/phi_etadash_k) * (-thdot_min + thdot + B2*u + mu_thdot -delta_bar ))**2 - sigma_thdot**2))
                sigma_k=np.min([sigma_k1,sigma_k2])
                if ex:
                    u = u + np.random.normal(0,sigma_k)
                u_type = "ex"
            else:
                u = - (thdot+mu_thdot)/B2 # u_stay
                # uppper and lower bounds of u_stay derived from conditions in Theorem 1
                u_stay_ub = (1/B2) * ( thdot_max - thdot - mu_thdot - delta_bar - phi_etadash_k*sigma_thdot)
                u_stay_lb = (1/B2) * ( thdot_min - thdot - mu_thdot + delta_bar + phi_etadash_k*sigma_thdot)
                if u_stay_ub < u_stay_lb:
                    raise ValueError('There is no conservative input "u_stay"')
                if u > u_stay_ub:
                    u=u_stay_ub
                elif u< u_stay_lb:
                    u=u_stay_lb                    
                u_type = "stay"
        else:
            u = - (thdot+ 2*mu_thdot)/B2   # u_back       
            # uppper and lower bounds of u_back derived from conditions in Theorem 1
            u_back_ub = (1/B2) * ( thdot_max - thdot -2*mu_thdot - Delta_bar - phi_xidash*np.sqrt(2)*sigma_thdot)
            u_back_lb = (1/B2) * ( thdot_min - thdot -2*mu_thdot + Delta_bar + phi_xidash*np.sqrt(2)*sigma_thdot)
            if u_back_ub < u_back_lb:
                raise ValueError('There is no conservative input "u_back"')
            if u > u_back_ub:
                u=u_back_ub
            elif u< u_back_lb:
                u=u_baclk_lb
            u_type = "back"
            
        return [u], u_type    

    def auto_u_adj_IFAC(self, u, step, ex): # proposed by Okawa et al., 2020 in IFAC World Congress
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        Ts = self.Ts
        u = u[0]
        B2= 3. *Ts/(m*l**2)
        delta_bar = 3. * g*Ts/ (2*l)
        delta_hat_bar = 2 * 3. * g*Ts/ (2*l)
        
        eta=self.eta
        tau=self.tau
        n_c=self.n_c
        
        eta_dash_0=1-(1-(eta)**(1/tau))/n_c
        phi_etadash_0 = norm.ppf(eta_dash_0, 0)               
        
        thdot_max=self.const_speed
        thdot_min=-self.const_speed 
        
        mu_thdot=self.mu_thdot
        sigma_thdot=self.sigma_thdot        
        
        if (thdot<=thdot_max) and (thdot>=thdot_min):
            if (sigma_thdot <= (1/phi_etadash_0) * (thdot_max - thdot - B2*u  -delta_bar ) )   \
            and (sigma_thdot <= (1/phi_etadash_0) * (-thdot_min + thdot + B2*u  - delta_bar ) ): 
                sigma_k1 = (1/ B2) * np.sqrt(  ( ((1/phi_etadash_0) * (thdot_max - thdot - B2*u  -delta_bar ))**2 ))
                sigma_k2 = (1/ B2) * np.sqrt(  ( ((1/phi_etadash_0) * (-thdot_min + thdot + B2*u  -delta_bar ))**2 ))
                sigma_k=np.min([sigma_k1,sigma_k2])
                if ex:
                    u = u + np.random.normal(0,sigma_k)
                u_type = "ex"
            else:
                u = - (thdot)/B2*0        
                u_type = "stay"
        else:
            u = - (thdot)/B2        
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
    
    res_u_all=[]
    res_th_all=[]
    res_thdot_all=[]

    for method_ID in [0,1]: 

        method=method_list[method_ID]
        print("Method: " + method)
    
        ActorNet_name=  load_path + header + "ActorNet_" + method  +"_Pend.h5"
        CriticNet_name=  load_path +  header + "CriticNet_" + method +"_Pend.h5"     

        env = Pend_ECMLPKDD2022(mu_th= mu_th, sigma_th= sigma_th, 
                               mu_thdot= mu_thdot, sigma_thdot= sigma_thdot)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        no_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0) * np.ones(1))
        
        actor_model = get_actor(num_states)
        critic_model = get_critic(num_states,num_actions)
        actor_model.load_weights(ActorNet_name)
        critic_model.load_weights(CriticNet_name)


        res_u_all.append([])
        res_th_all.append([])
        res_thdot_all.append([])


        for episode in range(1):
            prev_state = env.reset(ini_state=np.array([np.pi, 0]))
            res_th_all[method_ID].append(np.pi)
            res_thdot_all[method_ID].append(prev_state[2])
            for step in range( max_steps):
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = policy(tf_prev_state, no_noise,actor_model)
                if method == method_list[0]: 
                    action, u_type = env.auto_u_adj_IFAC(action, step, False)
                elif method ==method_list[1]:
                    action, u_type = env.auto_u_adj(action, step, False)
                state, reward, done, info, theta ,const_x = env.step(action)
                prev_state = state

                res_th_all[method_ID].append(theta)
                res_thdot_all[method_ID].append(prev_state[2])
                res_u_all[method_ID].append(float(action[0]))               


                if done:
                    break

            res_u_all[method_ID].append([])
            
        env.close()
        
        df_res_eval = pd.DataFrame(columns=['th', 'thdot', 'u'])  
        for step in range( max_steps+1):
            df_res_eval = df_res_eval.append({'th': res_th_all[method_ID][step], 
                                        'thdot': res_thdot_all[method_ID][step], 
                                        'u': res_u_all[method_ID][step]}, ignore_index=True)
        df_res_eval.to_csv(save_path + header +'res_eval_' +method + '_Pend.csv')


if __name__ == '__main__':
        
    main()

