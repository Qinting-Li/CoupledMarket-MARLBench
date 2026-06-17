"""Coupled electricity-carbon-green certificate market environment."""
import numpy as np
import copy
from config import *


class CoupledMarketEnv:

    def __init__(self, scenario=3, seed=42, agent_config=None):
        self.scenario = scenario
        self.agent_config = agent_config or {
            'nT': NUM_THERMAL, 'nR': NUM_RENEWABLE, 'nV': NUM_VPP,
            'n': NUM_AGENTS,
        }
        self._build_agents()
        self.num_agents = len(self.agent_specs)
        self.num_periods = NUM_PERIODS
        load_scale = max(float(self.agent_config.get('n', self.num_agents)) / NUM_AGENTS, 0.5)
        self.load = LOAD_PROFILE.copy() * load_scale

        self.thermal_ids = list(range(self.n_thermal))
        self.renewable_ids = list(range(self.n_thermal, self.n_thermal + self.n_renewable))
        self.vpp_ids = list(range(self.n_thermal + self.n_renewable, self.num_agents))

        self.state_dim = 5 + self.num_agents
        self.action_dim = 2

        self.reset()

    def _build_agents(self):
        self.n_thermal = int(self.agent_config.get('nT', NUM_THERMAL))
        self.n_renewable = int(self.agent_config.get('nR', NUM_RENEWABLE))
        self.n_vpp = int(self.agent_config.get('nV', NUM_VPP))

        self.agent_specs = []
        self.thermal_params = []
        self.renewable_params = []
        self.vpp_params = []

        for i in range(self.n_thermal):
            base = ['T1', 'T2'][i % 2]
            self.thermal_params.append(copy.deepcopy(THERMAL_PARAMS[base]))
            self.agent_specs.append({'type': 'thermal', 'name': f'T{i + 1}'})

        for i in range(self.n_renewable):
            base = ['WT', 'PV'][i % 2]
            self.renewable_params.append(copy.deepcopy(RENEWABLE_PARAMS[base]))
            self.agent_specs.append({'type': 'renewable', 'name': f'R{i + 1}'})

        for i in range(self.n_vpp):
            base = ['V1', 'V2'][i % 2]
            self.vpp_params.append(copy.deepcopy(VPP_PARAMS[base]))
            self.agent_specs.append({'type': 'vpp', 'name': f'V{i + 1}'})

    def reset(self):
        self.t = 0
        self.prev_outputs = np.zeros(self.num_agents)
        self.clearing_price = 180.0
        self.carbon_price = 60.0
        self.green_price = 40.0
        self.total_revenues = np.zeros(self.num_agents)

        self.soc = {}
        for i, vpp in enumerate(self.vpp_params):
            self.soc[i] = vpp['storage']['soc_init']

        self._load_renewable_actual()

        return self._get_state()

    def _load_renewable_actual(self):
        self.renewable_actual = []
        for params in self.renewable_params:
            actual = np.clip(params['profile'].copy(), 0, params['Q_max'])
            self.renewable_actual.append(actual)

    def _get_state(self):
        t = self.t % NUM_PERIODS
        state = np.zeros(self.state_dim)
        state[0] = self.load[t] / 600.0
        agent_end = 1 + self.num_agents
        state[1:agent_end] = self.prev_outputs / 200.0
        state[agent_end] = self.carbon_price / 100.0
        state[agent_end + 1] = self.green_price / 60.0
        state[agent_end + 2] = self.clearing_price / 300.0
        state[agent_end + 3] = t / 23.0
        return state

    def step(self, actions):
        t = self.t % NUM_PERIODS
        rewards = np.zeros(self.num_agents)
        info = {
            'clearing_price': 0, 'carbon_price': 0, 'green_price': 0,
            'outputs': np.zeros(self.num_agents),
            'costs': np.zeros(self.num_agents),
            'bids': np.zeros(self.num_agents),
            'quantities': np.zeros(self.num_agents),
            'vpp_components': {},
        }

        bids, quantities = self._decode_actions(actions, t)
        cleared_q, clearing_price = self._market_clearing(bids, quantities, t)
        self.clearing_price = clearing_price
        info['clearing_price'] = clearing_price
        info['outputs'] = cleared_q.copy()
        info['bids'] = bids.copy()
        info['quantities'] = quantities.copy()

        carbon_costs = np.zeros(self.num_agents)
        if self.scenario >= 3:
            self.carbon_price = self._calc_carbon_price(cleared_q, t)
            carbon_costs = self._calc_carbon_costs(cleared_q, t)
        info['carbon_price'] = self.carbon_price

        green_costs = np.zeros(self.num_agents)
        if self.scenario >= 2:
            self.green_price = self._calc_green_price(cleared_q, t)
            if self.scenario >= 3:
                self._carbon_green_linkage()
            green_costs = self._calc_green_costs(cleared_q, t)
        info['green_price'] = self.green_price

        for i in range(self.num_agents):
            if i in self.thermal_ids:
                rewards[i] = self._thermal_revenue(i, cleared_q[i],
                                                    clearing_price,
                                                    carbon_costs[i],
                                                    green_costs[i], t)
            elif i in self.renewable_ids:
                rewards[i] = self._renewable_revenue(i, cleared_q[i],
                                                      clearing_price,
                                                      green_costs[i], t)
            else:
                rewards[i] = self._vpp_revenue(i, cleared_q[i],
                                                clearing_price,
                                                carbon_costs[i],
                                                green_costs[i], t)
                vpp_idx = i - self.n_thermal - self.n_renewable
                info['vpp_components'][f'V{vpp_idx + 1}'] = (
                    self._last_vpp_components.copy())
            info['costs'][i] = carbon_costs[i] + green_costs[i]

        self.total_revenues += rewards
        self.prev_outputs = cleared_q.copy()
        self.t += 1
        done = (self.t >= NUM_PERIODS)
        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        return next_state, rewards, done, info

    def _decode_actions(self, actions, t):
        bids = np.zeros(self.num_agents)
        quantities = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            action = actions[i]
            q_ratio = np.clip((action[0] + 1) / 2, 0, 1)
            m_bid = np.clip(action[1] * 0.85 + 1.5, 0.8, 2.5)

            if i in self.thermal_ids:
                p = self.thermal_params[i]
                q = p['Q_min'] + q_ratio * (p['Q_max'] - p['Q_min'])
                if self.t > 0:
                    q = np.clip(q,
                                self.prev_outputs[i] - p['ramp_down'],
                                self.prev_outputs[i] + p['ramp_up'])
                q = np.clip(q, p['Q_min'], p['Q_max'])
                marginal_cost = p['a'] * q + p['b']
                if self.scenario >= 3:
                    marginal_cost += self.carbon_price * max(p['a_e'] - p['eta_e'], 0.0)
                if self.scenario >= 2:
                    marginal_cost += self.green_price * GREEN_CERT_MARKET['delta']
                bid = m_bid * marginal_cost
                quantities[i] = q
                bids[i] = bid

            elif i in self.renewable_ids:
                re_idx = i - self.n_thermal
                p = self.renewable_params[re_idx]
                q_max_t = p['profile'][t]
                q = q_ratio * q_max_t
                green_credit = self.green_price * (1 - GREEN_CERT_MARKET['delta']) if self.scenario >= 2 else 0.0
                bid = max(1.0, m_bid * 5.0 - 0.15 * green_credit)
                quantities[i] = q
                bids[i] = bid

            else:
                vpp_idx = i - self.n_thermal - self.n_renewable
                vpp = self.vpp_params[vpp_idx]
                gas_max = vpp['gas']['Q_max']
                wind_t = WIND_PROFILE[t] * vpp['wind']['profile_scale']
                solar_t = SOLAR_PROFILE[t] * vpp['solar']['profile_scale']
                re_total = wind_t + solar_t
                q_max = gas_max + re_total + vpp['storage']['P_max']
                q = q_ratio * q_max
                gas_q = min(max(q - re_total, vpp['gas']['Q_min']), gas_max)
                marginal_cost = vpp['gas']['a'] * gas_q + vpp['gas']['b']
                gas_ratio = gas_q / max(q, 1e-6)
                re_ratio = max(q - gas_q, 0.0) / max(q, 1e-6)
                if self.scenario >= 3:
                    marginal_cost += self.carbon_price * max(
                        vpp['gas']['a_e'] - vpp['gas']['eta_e'], 0.0) * gas_ratio
                if self.scenario >= 2:
                    marginal_cost -= 0.10 * self.green_price * (
                        1 - GREEN_CERT_MARKET['delta']) * re_ratio
                    marginal_cost += self.green_price * GREEN_CERT_MARKET['delta'] * gas_ratio
                marginal_cost = max(marginal_cost, 1.0)
                bid = m_bid * marginal_cost
                quantities[i] = q
                bids[i] = bid

        return bids, quantities

    def _market_clearing(self, bids, quantities, t):
        demand = self.load[t]
        order = np.argsort(bids)
        cleared_q = np.zeros(self.num_agents)
        remaining = demand

        for idx in order:
            if remaining <= 0:
                break
            alloc = min(quantities[idx], remaining)
            cleared_q[idx] = alloc
            remaining -= alloc

        clearing_price = 0.0
        for idx in order:
            if cleared_q[idx] > 0:
                clearing_price = bids[idx]

        unmet_ratio = max(remaining, 0.0) / max(demand, 1e-6)
        clearing_price = max(clearing_price, 20.0) + 250.0 * unmet_ratio

        return cleared_q, clearing_price

    def _calc_carbon_price(self, cleared_q, t):
        cm = CARBON_MARKET
        total_surplus = 0
        for i in self.thermal_ids:
            p = self.thermal_params[i]
            emissions = p['a_e'] * cleared_q[i]
            quota = p['eta_e'] * cleared_q[i]
            total_surplus += (quota - emissions)
        for i in self.vpp_ids:
            vpp_idx = i - self.n_thermal - self.n_renewable
            vpp = self.vpp_params[vpp_idx]
            re_out = (WIND_PROFILE[t] * vpp['wind']['profile_scale'] +
                      SOLAR_PROFILE[t] * vpp['solar']['profile_scale'])
            gas_q = max(cleared_q[i] - re_out, 0)
            gas_q = min(gas_q, vpp['gas']['Q_max'])
            emissions = vpp['gas']['a_e'] * gas_q
            quota = vpp['gas']['eta_e'] * gas_q
            total_surplus += (quota - emissions)

        price = cm['a_co2'] - cm['b_co2'] * total_surplus
        return np.clip(price, 20.0, 200.0)

    def _calc_carbon_costs(self, cleared_q, t):
        costs = np.zeros(self.num_agents)
        for i in self.thermal_ids:
            p = self.thermal_params[i]
            costs[i] = self.carbon_price * (p['a_e'] - p['eta_e']) * cleared_q[i]
        for i in self.vpp_ids:
            vpp_idx = i - self.n_thermal - self.n_renewable
            vpp = self.vpp_params[vpp_idx]
            re_out = (WIND_PROFILE[t] * vpp['wind']['profile_scale'] +
                      SOLAR_PROFILE[t] * vpp['solar']['profile_scale'])
            gas_q = max(cleared_q[i] - re_out, 0)
            gas_q = min(gas_q, vpp['gas']['Q_max'])
            costs[i] = self.carbon_price * (vpp['gas']['a_e'] - vpp['gas']['eta_e']) * gas_q
        return costs

    def _calc_green_price(self, cleared_q, t):
        gc = GREEN_CERT_MARKET
        delta = gc['delta']
        supply = 0
        for i in self.renewable_ids:
            supply += (1 - delta) * cleared_q[i]
        for i in self.vpp_ids:
            vpp_idx = i - self.n_thermal - self.n_renewable
            vpp = self.vpp_params[vpp_idx]
            re_out = (WIND_PROFILE[t] * vpp['wind']['profile_scale'] +
                      SOLAR_PROFILE[t] * vpp['solar']['profile_scale'])
            supply += (1 - delta) * re_out

        demand = 0
        for i in self.thermal_ids:
            demand += delta * cleared_q[i]

        price = (demand + gc['a_c'] - supply) / gc['b_c']
        return np.clip(price, 5.0, 150.0)

    def _calc_green_costs(self, cleared_q, t):
        costs = np.zeros(self.num_agents)
        gc = GREEN_CERT_MARKET
        delta = gc['delta']
        for i in self.thermal_ids:
            costs[i] = self.green_price * delta * cleared_q[i]

        for i in self.renewable_ids:
            costs[i] = -self.green_price * (1 - delta) * cleared_q[i]

        for i in self.vpp_ids:
            vpp_idx = i - self.n_thermal - self.n_renewable
            vpp = self.vpp_params[vpp_idx]
            re_out = (WIND_PROFILE[t] * vpp['wind']['profile_scale'] +
                      SOLAR_PROFILE[t] * vpp['solar']['profile_scale'])
            costs[i] = -self.green_price * (1 - delta) * re_out

        return costs

    def _thermal_revenue(self, agent_id, q, price, carbon_cost, green_cost, t):
        p = self.thermal_params[agent_id]
        gen_cost = 0.5 * p['a'] * q**2 + p['b'] * q + p['c']
        startup = 0
        if self.t > 0 and self.prev_outputs[agent_id] < p['Q_min'] + 1:
            startup = p['c'] * 0.1
        revenue = price * q - gen_cost - carbon_cost - green_cost - startup
        return revenue

    def _renewable_revenue(self, agent_id, q, price, green_cost, t):
        re_idx = agent_id - self.n_thermal
        p = self.renewable_params[re_idx]

        actual = self.renewable_actual[re_idx][t]
        predicted = q
        deviation = actual - predicted

        penalty = 0
        if deviation < 0:
            penalty = LAMBDA_PLUS * price * abs(deviation)
        else:
            penalty = LAMBDA_MINUS * price * deviation * 0.3

        revenue = price * q - green_cost - penalty
        return revenue

    def _vpp_revenue(self, agent_id, q, price, carbon_cost, green_cost, t):
        vpp_idx = agent_id - self.n_thermal - self.n_renewable
        vpp = self.vpp_params[vpp_idx]

        wind_t = WIND_PROFILE[t] * vpp['wind']['profile_scale']
        solar_t = SOLAR_PROFILE[t] * vpp['solar']['profile_scale']
        re_total = wind_t + solar_t

        gas_q = max(q - re_total, 0)
        gas_q = min(gas_q, vpp['gas']['Q_max'])
        gas_q = max(gas_q, 0)

        gas_cost = (0.5 * vpp['gas']['a'] * gas_q**2 +
                    vpp['gas']['b'] * gas_q + vpp['gas']['c'])
        if gas_q < 1.0:
            gas_cost = 0

        storage = vpp['storage']
        net_surplus = re_total + gas_q - q
        storage_cost = 0
        soc = self.soc[vpp_idx]
        storage_charge = 0
        storage_discharge = 0

        if net_surplus > 0:
            charge = min(net_surplus, storage['P_max'])
            charge = min(charge, (storage['soc_max'] - soc) * storage['capacity'])
            soc += charge * storage['eta_charge'] / storage['capacity']
            storage_cost = storage['lambda_charge'] * charge + storage['lambda_maint'] * charge
            storage_charge = charge
        elif net_surplus < 0:
            discharge = min(abs(net_surplus), storage['P_max'])
            discharge = min(discharge, (soc - storage['soc_min']) * storage['capacity'])
            soc -= discharge / (storage['eta_discharge'] * storage['capacity'])
            storage_cost = storage['lambda_discharge'] * discharge + storage['lambda_maint'] * discharge
            storage_discharge = discharge

        self.soc[vpp_idx] = np.clip(soc, storage['soc_min'], storage['soc_max'])
        re_used = min(re_total, q)
        if re_total > 1e-8:
            wind_used = re_used * wind_t / re_total
            solar_used = re_used * solar_t / re_total
        else:
            wind_used = 0.0
            solar_used = 0.0
        self._last_vpp_components = {
            'wind': float(wind_used),
            'solar': float(solar_used),
            'gas': float(gas_q),
            'storage_discharge': float(storage_discharge),
            'storage_charge': float(storage_charge),
            'cleared_q': float(q),
            'soc': float(self.soc[vpp_idx]),
        }

        revenue = price * q - gas_cost - storage_cost - carbon_cost - green_cost
        return revenue

    def _carbon_green_linkage(self):
        """Apply the carbon-green certificate price linkage."""
        carbon_base = CARBON_MARKET['a_co2']
        carbon_dev = (self.carbon_price - carbon_base) / carbon_base
        linkage_coeff = 0.15
        green_adjustment = 1.0 + linkage_coeff * carbon_dev
        self.green_price = np.clip(self.green_price * green_adjustment, 5.0, 150.0)

    def get_agent_obs(self, agent_id, state):
        return state.copy()

    def scenario_description(self):
        """Return benchmark-level metadata for the concrete energy scenario."""
        agent_roles = []
        for spec in self.agent_specs:
            if spec['type'] == 'thermal':
                role = 'ProducerAgent + carbon_emission'
            elif spec['type'] == 'renewable':
                role = 'ProducerAgent + certificate_generation'
            else:
                role = 'AggregatorAgent + storage + mixed_generation'
            agent_roles.append({
                'agent_id': spec['name'],
                'legacy_type': spec['type'],
                'benchmark_role': role,
                'markets': ['electricity', 'carbon', 'green_certificate'],
            })
        return {
            'scenario_name': 'electricity_carbon_green',
            'description': 'Electricity, carbon allowance, and green certificate coupled-market scenario.',
            'agents': agent_roles,
            'markets': ['electricity', 'carbon', 'green_certificate'],
            'resources': ['electricity', 'carbon_allowance', 'green_certificate', 'storage_energy'],
            'externalities': ['carbon_emission', 'renewable_certificate_shortfall'],
            'constraints': ['generation_capacity', 'ramping', 'renewable_availability', 'storage_soc', 'policy_quota'],
            'settlement': {
                'reward': (
                    'market_revenue - operating_cost - resource_cost - '
                    'externality_cost - violation_penalty + certificate_or_credit_revenue'
                )
            },
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'num_agents': self.num_agents,
        }

