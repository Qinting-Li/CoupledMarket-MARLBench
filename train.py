"""Training and experiment orchestration."""
import numpy as np
import time
import os
import copy
import pickle
from contextlib import contextmanager
from config import *
import config as config_module
import environment as env_module
from environment import CoupledMarketEnv
from algorithm import MultiAgentSystem


SWEEP_CACHE_VERSION = 8
REAL_SWEEP_EPISODES = 800


def _results_dir():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v3')
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path():
    return os.path.join(_results_dir(), '_real_sweep_cache.pkl')


def _load_sweep_cache():
    path = _cache_path()
    if not os.path.exists(path):
        return {'version': SWEEP_CACHE_VERSION, 'items': {}}
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    if cache.get('version') != SWEEP_CACHE_VERSION:
        return {'version': SWEEP_CACHE_VERSION, 'items': {}}
    cache.setdefault('items', {})
    return cache


def _save_sweep_cache(cache):
    with open(_cache_path(), 'wb') as f:
        pickle.dump(cache, f)


def _cache_get_or_train(key, trainer):
    cache = _load_sweep_cache()
    if key in cache['items']:
        print(f"    cache hit: {key}")
        return cache['items'][key]
    value = trainer()
    cache['items'][key] = value
    _save_sweep_cache(cache)
    return value


@contextmanager
def _temporary_market_params(**updates):
    original = {
        'thermal': copy.deepcopy(THERMAL_PARAMS),
        'renewable': copy.deepcopy(RENEWABLE_PARAMS),
        'vpp': copy.deepcopy(VPP_PARAMS),
        'carbon': copy.deepcopy(CARBON_MARKET),
        'green': copy.deepcopy(GREEN_CERT_MARKET),
        'lambda_plus_cfg': config_module.LAMBDA_PLUS,
        'lambda_minus_cfg': config_module.LAMBDA_MINUS,
        'lambda_plus_env': env_module.LAMBDA_PLUS,
        'lambda_minus_env': env_module.LAMBDA_MINUS,
        'load_cfg': config_module.LOAD_PROFILE.copy(),
        'load_env': env_module.LOAD_PROFILE.copy(),
        'wind_cfg': config_module.WIND_PROFILE.copy(),
        'wind_env': env_module.WIND_PROFILE.copy(),
        'solar_cfg': config_module.SOLAR_PROFILE.copy(),
        'solar_env': env_module.SOLAR_PROFILE.copy(),
    }
    try:
        if 'carbon_quota' in updates:
            for name in THERMAL_PARAMS:
                THERMAL_PARAMS[name]['eta_e'] = float(updates['carbon_quota'])
            for name in VPP_PARAMS:
                VPP_PARAMS[name]['gas']['eta_e'] = float(updates['carbon_quota']) * 0.72
        if 'green_quota' in updates:
            GREEN_CERT_MARKET['delta'] = float(updates['green_quota'])
        if 'lambda_plus' in updates:
            config_module.LAMBDA_PLUS = float(updates['lambda_plus'])
            env_module.LAMBDA_PLUS = float(updates['lambda_plus'])
        if 'lambda_minus' in updates:
            config_module.LAMBDA_MINUS = float(updates['lambda_minus'])
            env_module.LAMBDA_MINUS = float(updates['lambda_minus'])
        if 'carbon_price_base' in updates:
            CARBON_MARKET['a_co2'] = float(updates['carbon_price_base'])
            CARBON_MARKET['b_co2'] = 0.0
        if 'green_price_anchor' in updates:
            GREEN_CERT_MARKET['a_c'] = float(updates['green_price_anchor']) * GREEN_CERT_MARKET['b_c']
        yield
    finally:
        THERMAL_PARAMS.clear()
        THERMAL_PARAMS.update(original['thermal'])
        RENEWABLE_PARAMS.clear()
        RENEWABLE_PARAMS.update(original['renewable'])
        VPP_PARAMS.clear()
        VPP_PARAMS.update(original['vpp'])
        CARBON_MARKET.clear()
        CARBON_MARKET.update(original['carbon'])
        GREEN_CERT_MARKET.clear()
        GREEN_CERT_MARKET.update(original['green'])
        config_module.LAMBDA_PLUS = original['lambda_plus_cfg']
        config_module.LAMBDA_MINUS = original['lambda_minus_cfg']
        env_module.LAMBDA_PLUS = original['lambda_plus_env']
        env_module.LAMBDA_MINUS = original['lambda_minus_env']
        config_module.LOAD_PROFILE[:] = original['load_cfg']
        env_module.LOAD_PROFILE[:] = original['load_env']
        config_module.WIND_PROFILE[:] = original['wind_cfg']
        env_module.WIND_PROFILE[:] = original['wind_env']
        config_module.SOLAR_PROFILE[:] = original['solar_cfg']
        env_module.SOLAR_PROFILE[:] = original['solar_env']


def _train_sample(max_episodes, seed, algorithm='MA-CETD3', param_updates=None):
    param_updates = dict(param_updates or {})
    agent_config = param_updates.pop('agent_config', None)
    with _temporary_market_params(**param_updates):
        t0 = time.time()
        history, revenues, _ = train_scenario(
            3, algorithm, max_episodes=max_episodes, seed=seed,
            verbose=False, agent_config=agent_config)
        elapsed = time.time() - t0
    final_eval = history['final_eval']
    return {
        'history': history,
        'revenues': revenues,
        'mean_elec_price': float(np.mean(final_eval['clearing_prices'])),
        'mean_carbon_price': float(np.mean(final_eval['carbon_prices'])),
        'mean_green_price': float(np.mean(final_eval['green_prices'])),
        'final_reward': float(np.mean(history['total_rewards'][int(max_episodes * 0.8):])),
        'final_price': float(np.mean(history['clearing_prices'][int(max_episodes * 0.8):])),
        'time': float(elapsed),
    }


def _convergence_episode(rewards, threshold=0.95):
    rewards = np.array(rewards, dtype=float)
    if len(rewards) == 0:
        return 0
    tail = np.mean(rewards[int(len(rewards) * 0.8):])
    target = tail * threshold
    if tail >= 0:
        hits = np.where(rewards >= target)[0]
    else:
        hits = np.where(rewards >= target)[0]
    return int(hits[0]) if len(hits) else len(rewards)


def evaluate_trained_policy(scenario, mas, seed=42, agent_config=None):
    env = CoupledMarketEnv(scenario=scenario, seed=seed, agent_config=agent_config)
    state = env.reset()

    eval_data = {
        'outputs': [],
        'clearing_prices': [],
        'carbon_prices': [],
        'green_prices': [],
        'bids': [],
        'bid_quantities': [],
        'rewards': [],
        'vpp_components': {f'V{i + 1}': [] for i in range(env.n_vpp)},
    }

    for _ in range(NUM_PERIODS):
        actions = mas.select_actions(state, explore=False)
        next_state, rewards, done, info = env.step(actions)

        eval_data['outputs'].append(info['outputs'].tolist())
        eval_data['clearing_prices'].append(float(info['clearing_price']))
        eval_data['carbon_prices'].append(float(info['carbon_price']))
        eval_data['green_prices'].append(float(info['green_price']))
        eval_data['bids'].append(info['bids'].tolist())
        eval_data['bid_quantities'].append(info['quantities'].tolist())
        eval_data['rewards'].append(rewards.tolist())
        for name in eval_data['vpp_components']:
            eval_data['vpp_components'][name].append(info['vpp_components'][name])

        state = next_state
        if done:
            break

    eval_data['episode_rewards'] = np.sum(
        np.array(eval_data['rewards']), axis=0).tolist()
    return eval_data


def train_scenario(scenario, algorithm='MA-CETD3', max_episodes=None, seed=42,
                   verbose=True, agent_config=None):
    if max_episodes is None:
        max_episodes = ALGO_PARAMS['max_episodes']

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    env = CoupledMarketEnv(scenario=scenario, seed=seed, agent_config=agent_config)
    num_agents = env.num_agents
    mas = MultiAgentSystem(env.state_dim, env.action_dim, num_agents, algorithm)

    warmup_eps = min(ALGO_PARAMS['random_episodes'] // 3, max_episodes // 10)
    decay_eps = max_episodes * 0.45

    base_template = np.array([1.18, 1.08, 0.82, 0.78, 1.40, 1.50])
    range_template = np.array([0.15, 0.12, 0.10, 0.08, 0.22, 0.25])
    repeats = int(np.ceil(num_agents / len(base_template)))
    coeff_base = np.tile(base_template, repeats)[:num_agents]
    coeff_range = np.tile(range_template, repeats)[:num_agents]

    history = {
        'episode_rewards': [[] for _ in range(num_agents)],
        'total_rewards': [],
        'clearing_prices': [],
        'carbon_prices': [],
        'green_prices': [],
        'agent_outputs': [],
        'bidding_coeffs': [[] for _ in range(num_agents)],
    }

    for ep in range(max_episodes):
        state = env.reset()
        ep_rewards = np.zeros(num_agents)
        ep_prices = []
        ep_carbon = []
        ep_green = []
        ep_outputs = np.zeros((NUM_PERIODS, num_agents))
        ep_bids = np.zeros((NUM_PERIODS, num_agents))

        epsilon = max(0.08, 1.0 - ep / decay_eps) if ep < decay_eps else 0.08

        for t in range(NUM_PERIODS):
            if ep < warmup_eps or np.random.random() < epsilon:
                actions = {i: np.random.uniform(-1, 1, size=env.action_dim)
                           for i in range(num_agents)}
            else:
                actions = mas.select_actions(state, explore=True)

            next_state, rewards, done, info = env.step(actions)
            mas.store_transition(state, actions, rewards, next_state, done)

            ep_rewards += rewards
            ep_prices.append(info['clearing_price'])
            ep_carbon.append(info['carbon_price'])
            ep_green.append(info['green_price'])
            ep_outputs[t] = info['outputs']
            for i in range(num_agents):
                raw = actions[i][1] if env.action_dim > 1 else actions[i][0]
                ep_bids[t, i] = coeff_base[i] + raw * coeff_range[i]

            state = next_state

        if ep >= warmup_eps and len(mas.buffer) > ALGO_PARAMS.get('batch_size', 256):
            n_updates = min(4, max(1, int(4 * (1 - epsilon))))
            for _ in range(n_updates):
                mas.update()
            mas.decay_noise()
            for agent in mas.agents:
                if hasattr(agent, 'record_reward'):
                    agent.record_reward(float(np.sum(ep_rewards)))

        for i in range(num_agents):
            history['episode_rewards'][i].append(ep_rewards[i])
        history['total_rewards'].append(np.sum(ep_rewards))
        history['clearing_prices'].append(np.mean(ep_prices))
        history['carbon_prices'].append(np.mean(ep_carbon))
        history['green_prices'].append(np.mean(ep_green))
        history['agent_outputs'].append(np.sum(ep_outputs, axis=0).tolist())
        for i in range(num_agents):
            history['bidding_coeffs'][i].append(ep_bids[9, i])

        if verbose and (ep + 1) % 200 == 0:
            recent = history['total_rewards'][max(0, ep-99):ep+1]
            avg_r = np.mean(recent)
            print(f"  Episode {ep+1}/{max_episodes} | "
                  f"Avg Revenue: {avg_r:.0f} | Noise: {mas.noise_std:.4f}")

    final_start = max(0, int(max_episodes * 0.8))
    final_revenues = np.zeros(num_agents)
    for i in range(num_agents):
        final_revenues[i] = np.mean(history['episode_rewards'][i][final_start:])

    history['final_eval'] = evaluate_trained_policy(
        scenario, mas, seed + 10000, agent_config=agent_config)

    return history, final_revenues, mas


def run_all_scenarios(algorithm='MA-CETD3', max_episodes=None, seed=42):
    results = {}
    for sc in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Training Scenario {sc} with {algorithm}")
        print(f"{'='*60}")
        t0 = time.time()
        history, revenues, mas = train_scenario(sc, algorithm, max_episodes, seed)
        elapsed = time.time() - t0
        results[sc] = {'history': history, 'revenues': revenues, 'time': elapsed}
        print(f"Scenario {sc} done in {elapsed:.1f}s")
        names = ['T1', 'T2', 'WT', 'PV', 'V1', 'V2']
        for i, name in enumerate(names):
            print(f"  {name}: {revenues[i]:.0f} yuan")
        print(f"  Total: {np.sum(revenues):.0f} yuan")
    return results


def run_algorithm_comparison(max_episodes=None, seed=42):
    print("\n" + "="*60)
    print("Algorithm Comparison: MA-CETD3 vs MATD3")
    print("="*60)
    results = {}
    algo_seeds = {'MA-CETD3': seed, 'MATD3': seed + 7}
    for algo in ['MA-CETD3', 'MATD3']:
        print(f"\n--- {algo} ---")
        history, revenues, _ = train_scenario(3, algo, max_episodes, algo_seeds[algo])
        results[algo] = {'history': history, 'revenues': revenues}
    return results


def run_quota_analysis(max_episodes=None, seed=42):
    print("\n" + "="*60)
    print("Quota Coefficient Analysis")
    print("="*60)

    if max_episodes is None:
        max_episodes = REAL_SWEEP_EPISODES
    max_episodes = min(int(max_episodes), REAL_SWEEP_EPISODES)

    carbon_quotas = np.array([0.30, 0.45, 0.65, 0.85])
    green_quotas = np.array([0.15, 0.35, 0.55, 0.75])

    n_cq, n_gq = len(carbon_quotas), len(green_quotas)
    results = {
        'carbon_quotas': carbon_quotas.tolist(),
        'green_quotas': green_quotas.tolist(),
        'elec_prices': np.zeros((n_cq, n_gq)),
        'carbon_prices': np.zeros((n_cq, n_gq)),
        'green_prices': np.zeros((n_cq, n_gq)),
    }

    for i, cq in enumerate(carbon_quotas):
        for j, gq in enumerate(green_quotas):
            print(f"  Training quota point cq={cq:.2f}, gq={gq:.2f}")
            key = ('quota', int(max_episodes), seed, round(float(cq), 3), round(float(gq), 3))
            sample = _cache_get_or_train(
                key,
                lambda cq=cq, gq=gq, i=i, j=j: _train_sample(
                    max_episodes, seed + i * 100 + j,
                    param_updates={'carbon_quota': cq, 'green_quota': gq}))
            results['elec_prices'][i, j] = sample['mean_elec_price']
            results['carbon_prices'][i, j] = sample['mean_carbon_price']
            results['green_prices'][i, j] = sample['mean_green_price']

    print("  Quota analysis completed")
    return results


def run_penalty_analysis(max_episodes=None, seed=42):
    print("\n" + "="*60)
    print("Penalty Coefficient Analysis")
    print("="*60)

    if max_episodes is None:
        max_episodes = REAL_SWEEP_EPISODES
    max_episodes = min(int(max_episodes), REAL_SWEEP_EPISODES)

    lambda_plus_range = np.array([0.8, 1.1, 1.4, 1.8])
    lambda_minus_range = np.array([0.2, 0.6, 1.0, 1.4])

    n_lp, n_lm = len(lambda_plus_range), len(lambda_minus_range)
    results = {
        'lambda_plus': lambda_plus_range.tolist(),
        'lambda_minus': lambda_minus_range.tolist(),
        'elec_prices': np.zeros((n_lp, n_lm)),
        'carbon_prices': np.zeros((n_lp, n_lm)),
        'green_prices': np.zeros((n_lp, n_lm)),
    }

    for i, lp in enumerate(lambda_plus_range):
        for j, lm in enumerate(lambda_minus_range):
            print(f"  Training penalty point lambda_plus={lp:.2f}, lambda_minus={lm:.2f}")
            key = ('penalty', int(max_episodes), seed, round(float(lp), 3), round(float(lm), 3))
            sample = _cache_get_or_train(
                key,
                lambda lp=lp, lm=lm, i=i, j=j: _train_sample(
                    max_episodes, seed + 1000 + i * 100 + j,
                    param_updates={'lambda_plus': lp, 'lambda_minus': lm}))
            results['elec_prices'][i, j] = sample['mean_elec_price']
            results['carbon_prices'][i, j] = sample['mean_carbon_price']
            results['green_prices'][i, j] = sample['mean_green_price']

    print("  Penalty analysis completed")
    return results


def run_agent_count_analysis(seed=42):
    print("\n" + "="*60)
    print("Agent Count Convergence Analysis (3 groups)")
    print("="*60)
    max_eps = REAL_SWEEP_EPISODES

    group_thermal = [
        {'name': '1T-1R-2V', 'n': 4, 'nT': 1, 'nR': 1, 'nV': 2},
        {'name': '3T-1R-2V', 'n': 6, 'nT': 3, 'nR': 1, 'nV': 2},
    ]
    group_renewable = [
        {'name': '1T-1R-2V', 'n': 4, 'nT': 1, 'nR': 1, 'nV': 2},
        {'name': '1T-3R-2V', 'n': 6, 'nT': 1, 'nR': 3, 'nV': 2},
    ]
    group_vpp = [
        {'name': '1T-1R-2V', 'n': 4, 'nT': 1, 'nR': 1, 'nV': 2},
        {'name': '1T-3R-6V', 'n': 10, 'nT': 1, 'nR': 3, 'nV': 6},
    ]

    def gen_group_data(configs, group_seed):
        group = {}
        for idx, cfg in enumerate(configs):
            print(f"  Training agent configuration {cfg['name']}")
            key = ('agent_count', max_eps, group_seed, cfg['name'], cfg['nT'], cfg['nR'], cfg['nV'])
            sample = _cache_get_or_train(
                key,
                lambda cfg=cfg, idx=idx: _train_sample(
                    max_eps, group_seed + idx * 37 + cfg['n'],
                    param_updates={'agent_config': cfg}))
            rews = list(map(float, sample['history']['total_rewards']))
            prices = list(map(float, sample['history']['clearing_prices']))
            conv_ep = _convergence_episode(rews)
            group[cfg['name']] = {
                'rewards': rews, 'prices': prices,
                'n_agents': cfg['n'], 'conv_ep': conv_ep,
                'final_reward': float(np.mean(rews[-500:])),
                'final_price': float(np.mean(prices[-500:])),
                'nT': cfg['nT'], 'nR': cfg['nR'], 'nV': cfg['nV'],
            }
        return group

    results = {
        'group_thermal': gen_group_data(group_thermal, seed),
        'group_renewable': gen_group_data(group_renewable, seed + 10),
        'group_vpp': gen_group_data(group_vpp, seed + 20),
        'max_eps': max_eps,
    }
    print("  Agent count analysis (3 groups) completed")
    return results


def run_multi_algo_agent_comparison(seed=42):
    print("\n" + "="*60)
    print("Multi-Algorithm Agent Count Comparison")
    print("="*60)
    agent_counts = [4, 6, 10]
    algos = ['MA-CETD3', 'MATD3']
    max_eps = REAL_SWEEP_EPISODES
    results = {'agent_counts': agent_counts, 'algorithms': algos, 'data': {}}
    count_configs = {
        4: {'name': '4-agent-eq', 'n': 4, 'nT': 1, 'nR': 1, 'nV': 2},
        6: {'name': '6-agent-eq', 'n': 6, 'nT': 2, 'nR': 2, 'nV': 2},
        10: {'name': '10-agent-eq', 'n': 10, 'nT': 2, 'nR': 3, 'nV': 5},
    }
    for algo in algos:
        results['data'][algo] = {}
        for n in agent_counts:
            print(f"  Training {algo} with {n} agents")
            cfg = count_configs[n]
            key = ('multi_algo_agent', max_eps, seed, algo, n)
            sample = _cache_get_or_train(
                key,
                lambda algo=algo, cfg=cfg, n=n: _train_sample(
                    max_eps, seed + (0 if algo == 'MA-CETD3' else 5000) + n * 11,
                    algorithm=algo,
                    param_updates={'agent_config': cfg}))
            rews = list(map(float, sample['history']['total_rewards']))
            conv_ep = _convergence_episode(rews)
            results['data'][algo][n] = {
                'rewards': rews, 'conv_ep': conv_ep,
                'final_reward': float(np.mean(rews[-500:])),
                'training_time': float(sample.get('time', max_eps)),
            }
    print("  Multi-algo agent comparison completed")
    return results


def run_carbon_green_sensitivity(seed=42):
    print("\n" + "="*60)
    print("Carbon-Green Price Sensitivity Analysis")
    print("="*60)
    max_episodes = REAL_SWEEP_EPISODES
    carbon_ps = np.array([30, 55, 80, 110])
    green_ps = np.array([15, 35, 55, 75])
    results = {
        'carbon_prices': carbon_ps.tolist(),
        'green_prices': green_ps.tolist(),
        'elec_prices': np.zeros((len(carbon_ps), len(green_ps))),
    }
    for i, cp in enumerate(carbon_ps):
        for j, gp in enumerate(green_ps):
            print(f"  Training sensitivity point carbon={cp:.1f}, green_anchor={gp:.1f}")
            key = ('carbon_green_sensitivity', max_episodes, seed, int(cp), int(gp))
            sample = _cache_get_or_train(
                key,
                lambda cp=cp, gp=gp, i=i, j=j: _train_sample(
                    max_episodes, seed + 2000 + i * 100 + j,
                    param_updates={'carbon_price_base': cp, 'green_price_anchor': gp}))
            results['elec_prices'][i, j] = sample['mean_elec_price']
    print("  Carbon-green sensitivity completed")
    return results
