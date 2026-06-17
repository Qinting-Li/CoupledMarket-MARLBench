"""电碳绿证耦合市场参数配置"""

import numpy as np

NUM_PERIODS = 24
NUM_THERMAL = 2
NUM_RENEWABLE = 2
NUM_VPP = 2
NUM_AGENTS = NUM_THERMAL + NUM_RENEWABLE + NUM_VPP

# 负荷曲线 (MW)
LOAD_PROFILE = np.array([
    280, 260, 250, 245, 255, 290, 350, 420, 470, 490,
    500, 495, 480, 475, 460, 450, 470, 510, 520, 500,
    460, 400, 350, 310
], dtype=np.float64)

# 火电参数 C = 0.5*a*Q^2 + b*Q + c
THERMAL_PARAMS = {
    'T1': {
        'a': 0.0042, 'b': 18.0, 'c': 480.0,
        'Q_min': 30.0, 'Q_max': 200.0,
        'ramp_up': 40.0, 'ramp_down': 40.0,
        'a_e': 0.88, 'eta_e': 0.62,
        'bus': 1,
    },
    'T2': {
        'a': 0.0058,
        'b': 22.0,
        'c': 380.0,
        'Q_min': 20.0,
        'Q_max': 150.0,
        'ramp_up': 35.0,
        'ramp_down': 35.0,
        'a_e': 0.92,
        'eta_e': 0.58,
        'bus': 2,
    }
}

# 新能源参数
WIND_PROFILE = np.array([
    45, 50, 55, 60, 58, 52, 40, 35, 30, 28,
    25, 22, 20, 22, 28, 35, 42, 50, 55, 60,
    62, 58, 52, 48
], dtype=np.float64)

SOLAR_PROFILE = np.array([
    0, 0, 0, 0, 0, 5, 20, 40, 55, 65,
    72, 75, 73, 68, 58, 42, 25, 10, 2, 0,
    0, 0, 0, 0
], dtype=np.float64)

RENEWABLE_PARAMS = {
    'WT': {'Q_max': 100.0, 'profile': WIND_PROFILE, 'forecast_error_std': 0.08, 'bus': 10},
    'PV': {'Q_max': 80.0, 'profile': SOLAR_PROFILE, 'forecast_error_std': 0.10, 'bus': 30},
}

# 偏差惩罚系数
LAMBDA_PLUS = 1.2
LAMBDA_MINUS = 0.8

# 虚拟电厂参数
VPP_PARAMS = {
    'V1': {
        'gas': {'a': 0.0035, 'b': 28.0, 'c': 180.0,
                'Q_min': 10.0, 'Q_max': 55.0,
                'a_e': 0.52, 'eta_e': 0.45},
        'wind': {'Q_max': 45.0, 'profile_scale': 0.45},
        'solar': {'Q_max': 35.0, 'profile_scale': 0.44},
        'storage': {
            'capacity': 25.0,
            'P_max': 15.0,
            'eta_charge': 0.95,
            'eta_discharge': 0.95,
            'soc_min': 0.1,
            'soc_max': 0.9,
            'soc_init': 0.5,
            'lambda_charge': 8.0,
            'lambda_discharge': 10.0,
            'lambda_maint': 3.0,
        }
    },
    'V2': {
        'gas': {'a': 0.0032, 'b': 26.0, 'c': 200.0,
                'Q_min': 10.0, 'Q_max': 65.0,
                'a_e': 0.50, 'eta_e': 0.43},
        'wind': {'Q_max': 50.0, 'profile_scale': 0.50},
        'solar': {'Q_max': 40.0, 'profile_scale': 0.50},
        'storage': {
            'capacity': 30.0,
            'P_max': 18.0,
            'eta_charge': 0.95,
            'eta_discharge': 0.95,
            'soc_min': 0.1,
            'soc_max': 0.9,
            'soc_init': 0.5,
            'lambda_charge': 7.5,
            'lambda_discharge': 9.5,
            'lambda_maint': 2.8,
        }
    }
}

# 碳市场参数
CARBON_MARKET = {
    'a_co2': 95.0,
    'b_co2': 0.12,
}

# 绿证市场参数
GREEN_CERT_MARKET = {
    'delta': 0.30,
    'a_c': 55.0,
    'b_c': 0.25,
}

# 算法参数
ALGO_PARAMS = {
    'lr_actor': 0.001,
    'lr_critic': 0.001,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'buffer_size': 100000,
    'hidden_dims': [256, 128, 64],
    'noise_std': 0.2,
    'noise_decay': 0.999,
    'policy_delay': 2,
    'noise_clip': 0.5,
    'max_episodes': 4000,
    'random_episodes': 2000,
    'cem_pop_size': 50,
    'cem_elite_frac': 0.2,
    'cem_sigma_init': 1.0,
    'cem_beta': 0.1,
    'cem_sigma_min': 0.01,
    'lambda_init': 0.7,
    'lambda_decay': 0.998,
}

# 训练参数
TRAIN_PARAMS = {
    'seed': 42,
    'eval_interval': 100,
    'save_interval': 500,
    'num_scenarios': 3,
}
