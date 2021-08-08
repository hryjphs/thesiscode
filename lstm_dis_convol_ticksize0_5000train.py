# -*- coding: utf-8 -*-
"""
lstm_discrete_constant vol 
c : ticksize to control the cost -- 0.1
"""

"""
Initial time to maturity is 5 days; Assume we could trade 5 times a day; short 1 European call option contract ( each for 100 shares)
Geometric brownian simulation (constant volatility) or SABR (stochastic volatility) to simulate the underlying stock prices
Action space: the holding position of the stock: [0,100], discrete
State space: [current stock price, current holding postion, time to maturity,lstm_re]
Reward: profit and loss
RL algorithm:  Double dueling DQN with prioritized  replay  experience-->stablize, avoid overestimation, speed up

"""

###########################################
###############   Simulation  #############
###########################################
# -*- coding: utf-8 -*-
"""
To simulate paths for the underlying stock and European call option with constant volatility and stochastic volatility;
Constant volatility: BSM delta is calculated (benchmark)
Stochastic volatility: SABR model by Hagen et al (2002) is used. “Practitioner delta" and  “Bartlett’s delta” (Bartlett, 2006)
are calculated (benchmark)

H defines how many days needed as history (H/frequency --> number of previuos trades go into lstm layer to get features)

"""

import random
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')

random.seed(1)

#############################
####   Constant vol  ########
#############################

def brownian_sim(num_path, num_period, mu, std, init_p, dt):
    """
    Assume dSt = St (mu*dt + std*dWt), where Wt is brownian motion
    Input: num_path: number of path to simulate; num_period: the length of a path; init_p: initial price
    Return un_price, the underlying stock price
    """
    z = np.random.normal(size=(num_path,num_period))
    
    un_price = np.zeros((num_path,num_period))
    un_price[:,0] = init_p
    
    for t in range(num_period-1):
        un_price[:,t+1] = un_price[:,t] * np.exp((mu - (std ** 2)/ 2)* dt + std * np.sqrt(dt) * z[:,t])
    
    return un_price



def bs_call(iv, T, S, K, r, q):
    """
    BSM Call Option Pricing Formula & BS Delta formula 
    Input: T here is time to maturity, iv : implied volatility, q : continuous dividend,
            r : risk free rate, S : current stock price, K : strike price
    Return bs_price, BSM call option price;  bs_delta, BSM delta
    """
    
    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_delta = np.exp(-q * T) * norm.cdf(d1)
    return bs_price, bs_delta


def get_sim_path(M, freq, np_seed, num_sim, H=0, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0):                                                                
    """ 
    Simulate paths
    Input: M: initial time to maturity, days; H: how many days before should be fed into LSTM; freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
            np_seed: numpy random seed; num_sim: number of simulation path; mu: annual return; vol: annual volatility
            S: initial asset value; K: strike price; r: annual risk free rate; q: annual dividend
            If risk-neutrality, mu = r-q
    Return simulated data: a tuple of three arrays
        1) asset price paths (num_path x num_path x [num_period+num_history])
        2) option price paths (num_path x num_period)
        3) delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)
    
    # Annual Trading Day
    T = 250

    # Change into unit of year
    dt = 0.004 * freq

    # Number of period
    num_period = int((M+H) / freq)
    num_history = int(H / freq)

    # underlying asset price 2-d array
    print("1. generate asset price paths")
    un_price = brownian_sim(num_sim, num_period + 1, mu, vol, S, dt)

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, 0, -freq) #np.arrage(start,stop,step) from  [start,stop)
    ttm = np.append(ttm,0)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price and delta")
    bs_price, bs_delta = bs_call(vol, ttm / T, un_price[:,num_history:], K, r, q)  # bs_call(iv, T, S, K, r, q)

    print("simulation done!")

    return un_price, bs_price, bs_delta

#############################
####   Stochastic vol  ######
#############################

def sabr_sim(num_path, num_period, mu, std, init_p, dt, rho, beta, volvol):
    """
     We assume an extension of geometric Brownian motion where the volatility is stochastic : dS =µSdt+σSdz_1  ;  dσ =vσdz_2
     Input: rho: the constant correlation between dz_1 and dz_2, two Wiener processes
             volvol: the volatility of volatility process, std : initial volatility
     Return a_price, underlying asset price path; vol, the volatility path
    """
    qs = np.random.normal(size=(num_path, num_period))
    qi = np.random.normal(size=(num_path, num_period))
    qv = rho * qs + np.sqrt(1 - rho * rho) * qi   #sum of normal is normal --> construct a wiener process dz2 with correlation rho 

    vol = np.zeros((num_path, num_period))
    vol[:, 0] = std

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        gvol = vol[:, t] * (a_price[:, t] ** (beta - 1))  #beta = 1 
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (gvol ** 2) / 2) * dt + gvol * np.sqrt(dt) * qs[:, t]
        )
        vol[:, t + 1] = vol[:, t] * np.exp(
            -volvol * volvol * 0.5 * dt + volvol * qv[:, t] * np.sqrt(dt)
        )

    return a_price, vol


def sabr_implied_vol(vol, T, S, K, r, q, beta, volvol, rho):
    """ 
    Input: vol is initial volatility, T time to maturity
    Return implied volatility  SABRIV
    """

    F = S * np.exp((r - q) * T)
    x = (F * K) ** ((1 - beta) / 2)
    y = (1 - beta) * np.log(F / K)
    A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920))
    B = 1 + T * (
        ((1 - beta) ** 2) * (vol * vol) / (24 * x * x)
        + rho * beta * volvol * vol / (4 * x)
        + volvol * volvol * (2 - 3 * rho * rho) / 24
    )
    Phi = (volvol * x / vol) * np.log(F / K)
    Chi = np.log((np.sqrt(1 - 2 * rho * Phi + Phi * Phi) + Phi - rho) / (1 - rho))

    SABRIV = np.where(F == K, vol * B / (F ** (1 - beta)), A * B * Phi / Chi)

    return SABRIV


def bartlett(sigma, T, S, K, r, q, ds, beta, volvol, rho): 
    """
    Return barlett delta
    """

    dsigma = ds * volvol * rho / (S ** beta)

    vol1 = sabr_implied_vol(sigma, T, S, K, r, q, beta, volvol, rho)  #sabr_implied_vol(vol, T, S, K, r, q, beta, volvol, rho): sigma here is initial volatility
    vol2 = sabr_implied_vol(sigma + dsigma, T, S + ds, K, r, q, beta, volvol, rho)

    bs_price1, _ = bs_call(vol1, T, S, K, r, q)
    bs_price2, _ = bs_call(vol2, T, S+ds, K, r, q)

    b_delta = (bs_price2 - bs_price1) / ds

    return b_delta


def get_sim_path_sabr(M, freq, np_seed, num_sim, H =0, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0, beta=1, rho=-0.4, volvol = 0.6, ds = 0.001):
    """ 
        Input: M: initial time to maturity; H: how many days before should be fed into LSTM; freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
            np_seed: numpy random seed; num_sim: number of simulation path; 
        Return simulated data: a tuple of four arrays
            1) asset price paths (num_path x [num_period+num_history])
            2) option price paths (num_path x num_period)
            3) bs delta (num_path x num_period)
            4) bartlett delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)

    # Annual Trading Day
    T = 250

    # Change into unit of year
    dt = 0.004 * freq

    # Number of period
    num_period = int((M+H) / freq)
    num_history = int(H / freq)


    # asset price 2-d array; sabr_vol
    print("1. generate asset price paths (sabr)")
    a_price, sabr_vol = sabr_sim(
        num_sim, num_period + 1, mu, vol, S, dt, rho, beta, volvol
    )

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, 0, -freq) #np.arrage(start,stop,step) from  [start,stop)
    ttm = np.append(ttm,0)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price, BS delta, and Bartlett delta")

    # sabr implied vol
    implied_vol = sabr_implied_vol(
        sabr_vol[:,num_history:], ttm / T, a_price[:,num_history:], K, r, q, beta, volvol, rho
    )

    bs_price, bs_delta = bs_call(implied_vol, ttm / T, a_price[:,num_history:], K, r, q)

    bartlett_delta = bartlett(sabr_vol[:,num_history:], ttm / T, a_price[:,num_history:], K, r, q, ds, beta, volvol, rho)

    print("simulation done!")

    return a_price, bs_price, bs_delta, bartlett_delta

###########################################
###############   schedule      ###########
###########################################

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

"""memory"""

import numpy as np
import random
import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """ save data to memory"""
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize  #the pointer go forward, when full go back to the initial to overwrite

    def _encode_sample(self, idxes):
        """ retrieve data from memory"""
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.  
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)] #random.randint [a,b] , it's with replacement!
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        """ sample idxes"""
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights (to reduce bias)
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

###########################################
###############   environment      ########
###########################################
"""new env"""

"""
A new trading environment, which contains
(1) observation space: already simulated samples 
(2) action space: continuous or discrete, holdings of the hedging assets
(3) reset(): go to a new episode
(4) step(): transist to the next state

Now it also considers the intraday volumne difference --> different price impact: assume trade 5 times a day, multi_t [2,1.5,1,1.5,2]

"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class TradingEnv(gym.Env):
    """
    trading environment;
    contains observation space (already simulated samples), action space, reset(), step()
    """

    def __init__(self, cash_flow_flag=0, dg_random_seed=1, num_sim=500002, sabr_flag = False,
        continuous_action_flag=False, ticksize=0, init_ttm=5,history=5, trade_freq=0.2, num_contract=1, sim_flag = True,
        mu=0.05, vol=0.2, S=100, K=100, r=0, q=0, beta=1, rho=-0.4, volvol = 0.6, ds = 0.001, k=0):
        """ cash_flow_flag: 1 if reward is defined using cash flow, 0 if profit and loss; dg_random_seed: random seed for simulation;
            num_sim: number of paths to simulate; sabr_flag: whether use sabr model; continuous_action_flag: continuous or discrete
            action space; init_ttm: initial time to maturity in unit of day; history:how many days needed as history in lstm layer;
            trade_freq: trading frequency in unit of day;
            num_contract: number of call option to short; sim_flag = simulation or market data;
            Assume the trading cost include spread cost and price impact cost, cost(n)= multi_t*ticksize*(|n|+0.01*n^2)
            Reward is about ((change in value) - k/2 * (change in value)^2 ), where k is risk attitude, k=0 if risk neutral
        """
        # observation
        if sim_flag:
            # generate data now
            if sabr_flag:
                self.path, self.option_price_path, self.delta_path, self.bartlett_delta_path = get_sim_path_sabr(M=init_ttm, freq=trade_freq, 
                                                                                                                 np_seed=dg_random_seed, num_sim=num_sim, H=history, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0,
                                                                                                                  beta=1, rho=-0.4, volvol = 0.6, ds = 0.001)
            
            else:
                self.path, self.option_price_path, self.delta_path = get_sim_path(M=init_ttm, freq=trade_freq,
                                                                                  np_seed=dg_random_seed, num_sim=num_sim, H=history, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0)
        else:
            # use actual data ---> to be continued
            return 
        
        # other attributes
        self.num_path = self.path.shape[0]

        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in simulation.py): (s0,s1...sT) -->T+1
        self.num_period = self.option_price_path.shape[1]
        # print("***", self.num_period)

        # time to maturity array
        self.ttm_array = np.arange(init_ttm, 0, -trade_freq)
        self.ttm_array = np.append(self.ttm_array,0)
        # print(self.ttm_array)
        
        # time of a day [1,2,3,4,5,1,2,3,4,5.....] corresponds to [2,1.5,1,1.5,2.....]
        self.time_of_day = [2,1.5,1,1.5,2]*init_ttm

        # cost part
        self.ticksize = ticksize  
        #self.multi = multi

        # risk attitude
        self.k = k   

        # history
        self.history_trades = int(history/trade_freq)

        #self.trad_freq = trad_freq
        #int(self.history/self.trade_freq)                                                                              

        # step function initialization depending on cash_flow_flag
        if cash_flow_flag == 1:
            self.step = self.step_cash_flow   # see step_cash_flow() definition below. Internal reference use self.
        else:
            self.step = self.step_profit_loss

        self.num_contract = num_contract
        self.strike_price = 100

        # track the index of simulated path in use
        self.sim_episode = -1

        # track time step within an episode (it's step)
        self.t = 0

        # action space for holding 
        # With L contracts, each for 100 shares, one would not want to trade more than 100·L shares                                                                                                       #action space justify?
        if continuous_action_flag:
            self.action_space = spaces.Box(low=np.array([0]), high=np.array([num_contract * 100]), dtype=np.float32)
        else:
            self.num_action = num_contract * 100 + 1
            self.action_space = spaces.Discrete(self.num_action)  #number from 0 to self.num_action-1

        # state element, assumed to be the sum: current price, current holding, ttm, lstm inputs:history/trade_freq -->3+25=28
        self.num_state = 28

        self.state = []    # initialize current state

        # seed and start
        self.seed()  # call this function when intialize ...
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)  #self.np_random now is a generateor np.random.RandomState() with a strong random seed; seed is a strong random seed
        return [seed]

    def reset(self):
        # repeatedly go through available simulated paths (if needed)
        # start a new episode
        self.sim_episode = (self.sim_episode + 1) % self.num_path

        self.t = 0

        price = self.path[self.sim_episode, self.history_trades ]
        position = 0

        ttm = self.ttm_array[self.t]
        
        history = list(self.path[self.sim_episode, 0:self.history_trades ])

        self.state = [price, position, ttm] + history  # history as a list
        return self.state

    def step_cash_flow(self, action):
        """
        cash flow period reward
        take a step and return self.state, reward, done, info
        """

        # do it consistently as in the profit & loss case
        # current prices (at t)
        current_price = self.state[0]

        # current position
        current_position = self.state[1]
        
         #get the multi_t
        multi_t = self.time_of_day[self.t]

        # update time/period
        self.t = self.t + 1

        # get state for tomorrow
        price = self.path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]
        history = list(self.path[self.sim_episode, self.t:(self.history_trades+self.t) ])

        self.state = [price, position, ttm] + history   #state transist to next price, ttm and stores current action(position)
        
       
        
        # calculate period reward (part 1)
        cash_flow = -(position - current_position) * current_price - (np.abs(position - current_position) +0.01*(position - current_position)**2) * self.ticksize * multi_t    

        # if tomorrow is end of episode, only when at the end day done=True , self.num_period = T/frequency +1
        if self.t == self.num_period - 1:
            done = True   #you have arrived at the terminal
            # add (stock payoff + option payoff) to cash flow
            cash_flow = cash_flow + price * position - max(price - self.strike_price, 0) * self.num_contract * 100 - (position + 0.01*position**2) * self.ticksize * multi_t  
            reward = cash_flow -self.k /2 * (cash_flow)**2
        else:
            done = False
            reward = cash_flow -self.k /2 * (cash_flow)**2

        # for other info
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info

    def step_profit_loss(self, action):
        """
        profit loss period reward
        """

        # current prices (at t)
        current_price = self.state[0]
        current_option_price = self.option_price_path[self.sim_episode, self.t]

        # current position
        current_position = self.state[1]
        
         #get the multi_t
        multi_t = self.time_of_day[self.t]

        # update time
        self.t = self.t + 1

        # get state for tomorrow
        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]
        history = list(self.path[self.sim_episode, self.t:(self.history_trades+self.t) ])

        self.state = [price, position, ttm] + history   #state transist to next price, ttm and stores current action(position)
        


        # calculate period reward (part 1)
        reward = (price - current_price) * position - (np.abs(position - current_position) +0.01*(position - current_position)**2) * self.ticksize * multi_t 

        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) * self.num_contract * 100 - (position + 0.01*position**2) * self.ticksize * multi_t   #liquidate option and stocks
            reward = reward - self.k / 2 * (reward)**2
        else:
            done = False
            reward = reward - (option_price - current_option_price) * self.num_contract * 100
            reward = reward - self.k / 2 * (reward)**2
        # for other info later
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info

###########################################
###############   Agent      #############
###########################################
"""
Double dueling DQN+PRE

"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc0_dims, fc1_dims, fc2_dims, name, chkpt_dir='tmp/ddqn'):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.lstm = keras.layers.LSTM(fc0_dims,input_shape=(25,1))   # add lstm
        self.V = keras.layers.Dense(1, activation=None)  #linear
        self.A = keras.layers.Dense(n_actions, activation=None)
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddqn')

    def call(self, state):
        #state = keras.layers.BatchNormalization(state)
        t = self.lstm(state[:,3:])
        t = tf.expand_dims(t,-1)
        x = tf.squeeze(tf.concat([t,state[:,0:3]],1))
        #keras.layers.BatchNormalization(x)
        x = self.dense1(x)
        #keras.layers.BatchNormalization(x)
        x = self.dense2(x)
        #keras.layers.BatchNormalization(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        #state = keras.layers.BatchNormalization(state)
        t = self.lstm(state[:,3:])
        t = tf.expand_dims(t,-1)
        x = tf.reshape(tf.concat([t,state[:,0:3]],1),[1,8])
        #keras.layers.BatchNormalization(x)
        x = self.dense1(x)
        #keras.layers.BatchNormalization(x)
        x = self.dense2(x)
        #keras.layers.BatchNormalization(x)
        A = self.A(x)

        return A

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, env,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=100000, fc0_dims = 5, fc1_dims=128,
                 fc2_dims=128, replace=100, prioritized_replay= False):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay
        self.learn_step_counter = 0
        self.t = env.t
        
        self.q_eval = DuelingDeepQNetwork(n_actions, fc0_dims, fc1_dims, fc2_dims, name='eval_dc0')
        self.q_next = DuelingDeepQNetwork(n_actions, fc0_dims, fc1_dims, fc2_dims, name='target_dc0')

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        
        if self.prioritized_replay == False:
            self.memory = ReplayBuffer(mem_size)
        else:
            # memory buffer for experience replay
            prioritized_replay_alpha = 0.6
            self.memory = PrioritizedReplayBuffer(mem_size, alpha=prioritized_replay_alpha)
            prioritized_replay_beta0 = 0.4
            # need not be the same as training episode 
            prioritized_replay_beta_iters = 50001

            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)

            # for numerical stabiligy
            self.prioritized_replay_eps = 1e-6
        
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.add(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action
    
    def greedy_action(self,observation):
        
        state = np.array([observation])
        actions = self.q_eval.advantage(state)
        action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    
    
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        
        if self.prioritized_replay == False:
            states, actions, rewards, states_, dones = \
                                    self.memory.sample(self.batch_size)
            actions = actions.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)
            
            q_pred = self.q_eval(states)
            q_next = self.q_next(states_)
            # changing q_pred doesn't matter because we are passing states to the train function anyway
            # also, no obvious way to copy tensors in tf2?
            q_target = q_pred.numpy()
            max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
            
            for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
                q_target[idx, actions[idx]] = rewards[idx] + \
                        self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
            self.q_eval.train_on_batch(states, q_target)

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

            self.learn_step_counter += 1
            
        else:
        
            # prioritized sample from experience replay buffer
            experience = self.memory.sample(self.batch_size, beta=self.beta_schedule.value(self.t))
            (states, actions, rewards, states_, dones, weights, batch_idxes) = experience

            states = tf.convert_to_tensor(np.expand_dims(states, axis = -1))
            states_ = tf.convert_to_tensor(np.expand_dims(states_, axis = -1))

            actions = actions.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)
            weights = weights.flatten()

            q_pred = self.q_eval(states) #eval
            q_next = self.q_next(states_) #target
            # changing q_pred doesn't matter because we are passing states to the train function anyway
            # also, no obvious way to copy tensors in tf2?
            q_target = q_pred.numpy()
            max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
            q_tar =[]
            q_pre =[]
            for idx, terminal in enumerate(dones):
                q_target_ =  rewards[idx] + \
                        self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
                q_target[idx, actions[idx]] = q_target_
                q_tar.append(q_target_.numpy()[0])
                q_pre_ = q_pred[idx,actions[idx][0]].numpy()
                q_pre.append(q_pre_)
                
            self.q_eval.train_on_batch(states, q_target,sample_weight=weights)
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

            self.learn_step_counter += 1       
            # use Q2 TD error as priority weight
            td_errors = np.array(q_pre) - np.array(q_tar)
            new_priorities = (np.abs(td_errors) + self.prioritized_replay_eps).flatten()
            self.memory.update_priorities(batch_idxes, new_priorities)
        



    
    def save_model(self):
        print('... saving models ...')
        self.q_eval.save_weights(self.q_eval.checkpoint_file)
        self.q_next.save_weights(self.q_next.checkpoint_file)
        
        
    def load_model(self):
        print('... loading models ...')
        if os.path.exists(self.q_eval.checkpoint_file):
            self.q_eval.load_weights(self.q_eval.checkpoint_file)
            print('...successfully load q_eval...')
        if os.path.exists(self.q_next.checkpoint_file):
            self.q_next.load_weights(self.q_next.checkpoint_file)
            print('...successfully load q_next...')
###########################################
#######  plots and comparison      ########
###########################################
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.clf()  
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def plot_obj(scores, figure_file):
    plt.clf()
    plt.hist(scores)
    plt.savefig(figure_file)

def test(total_episode_test, env, agent, name, delta_flag=False, bartlett_flag=False):
        """
        hedge with model: RL/BSM
        report the objectives for every 1000 episode
        save the objectives for every episode to csv 
        
        """
        print('testing...')
        
        agent.load_model()

        u_T_store = []

        for i in range(total_episode_test):
            observation = env.reset()
            done = False
            action_store = []
            reward_store = []

            while not done:

                # prepare state
                #x = np.array(observation).reshape(1, -1)

                if delta_flag:
                    action = env.delta_path[i % env.num_path, env.t] * env.num_contract * 100
                elif bartlett_flag:
                    action = env.bartlett_delta_path[i % env.num_path, env.t] * env.num_contract * 100
                else:
                    # choose action from greedy                                 
                    action = agent.greedy_action(tf.convert_to_tensor(np.expand_dims(observation,-1)))

                # store action to take a look
                action_store.append(action)

                # a step
                observation, reward, done, info = env.step(action)
                reward_store.append(reward)

            # get final utility at the end of episode, and store it.
            u_T = sum(reward_store)
            u_T_store.append(u_T)

            if i % 1000 == 0:
                u_T_mean = np.mean(u_T_store)
                u_T_var = np.var(u_T_store)
                path_row = info["path_row"]
                print(info)
                with np.printoptions(precision=2, suppress=True):
                    print("episode: {} | final utility Y(0): {:.2f}; so far mean and variance of final utility was {} and {}".format(i, u_T, u_T_mean, u_T_var))       
                    print("episode: {} | rewards: {}".format(i, np.array(reward_store)))
                    print("episode: {} | action taken: {}".format(i, np.array(action_store)))
                    print("episode: {} | deltas {}".format(i, env.delta_path[path_row] * 100))  
                    print("episode: {} | stock price {}".format(i, env.path[path_row]))
                    print("episode: {} | option price {}\n".format(i, env.option_price_path[path_row] * 100))
        
        upperbound = total_episode_test + 1
        epi = np.arange(1, upperbound, 1)  
        history = dict(zip(epi, u_T_store))
        #name = os.path.join('history', name)
        df = pd.DataFrame(history,index=[0])
        df.to_csv(name, index=False, encoding='utf-8')
        
        return u_T_store

###########################################
####    train & test -constant vol      ###
###########################################
"""
Agent interact with the environment; based on memory, train the model
"""


import numpy as np
import gym
#from utils import plotLearning
import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    #tf.compat.v1.disable_eager_execution()
    num_simulation = 5000
    env = TradingEnv(num_sim = num_simulation, continuous_action_flag=False) # see tradingenv.py for more info 
    lr = 0.001
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, n_actions = 101,
                input_dims=env.num_state,env = env,
                mem_size=1000, batch_size=128,
                 prioritized_replay= True)
   
    agent.load_model()

    scores = []
    #eps_history = []


    for i in range(num_simulation):
        #interaction
        done = False
        score = 0
        observation = env.reset()  #[price, position, ttm], price=S, position=0, ttm=init_ttm
        j=0
        while not done:
            action = agent.choose_action(tf.convert_to_tensor(np.expand_dims(observation,-1)))  #action is tensor
            #action = action.numpy()[0]           #change to numpy
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

            

        #eps_history.append(agent.epsilon)
        scores.append(score)  # score for every episode

        avg_score = np.mean(scores[-100:])
        if i % 100 == 0:
          print('episode %.2f' % i, 'score %.2f' % score, 'average_score %.2f' % avg_score)
        #        'epsilon %.2f' % agent.epsilon)

    filename = 'dddqn_tf2_lstm_dc0.png'
    x = [i+1 for i in range(num_simulation)]
    plot_learning_curve(x, scores, filename)
    agent.save_model()



    total_episode_test = 3000
    env_test2 = TradingEnv(continuous_action_flag=False, sabr_flag=False, dg_random_seed=2, num_sim=total_episode_test)

    delta_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='delta_dc0', delta_flag=True, bartlett_flag=False)
    rl_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='rl_dc0', delta_flag=False, bartlett_flag=False)
    
    plot_obj(delta_u, figure_file='delta_u_dc0')
    plot_obj(rl_u, figure_file='rl_u_dc0')











