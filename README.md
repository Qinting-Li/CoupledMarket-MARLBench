# CoupledMarket-MARLBench

A general multi-agent reinforcement learning benchmark for coupled markets with
heterogeneous agents, multi-layer settlement, externality pricing, and
operational constraints.

The original electricity-carbon-green certificate market is now one scenario,
not the whole project. The benchmark is organized around reusable abstractions:

- **Agents**: heterogeneous decision makers.
- **Markets**: coupled trading layers.
- **Resources**: tradable commodities, services, credits, and capacities.
- **Externalities**: emissions, risk, congestion, pollution, reliability, and other priced effects.
- **Constraints**: physical, budget, policy, and safety limits.
- **Settlement**: multi-market accounting and reward construction.

The generic reward ledger is:

```text
reward_i =
  market_revenue
  - operating_cost
  - resource_cost
  - externality_cost
  - violation_penalty
  + certificate_or_credit_revenue
```

## Project Layout

```text
market_env/
  core/
    agent.py
    market.py
    resource.py
    constraint.py
    settlement.py
    simulator.py
  scenarios/
    electricity_carbon_green/
    cloud_energy_carbon/
    logistics_carbon_credit/
    edge_computing_energy/
    traffic_energy_carbon/
algorithm.py
train.py
visualize.py
main.py
```

## Built-in Scenarios

| Scenario | Coupled markets |
| --- | --- |
| `electricity_carbon_green` | electricity market + carbon market + green certificate market |
| `cloud_energy_carbon` | compute market + electricity price + carbon cost |
| `traffic_energy_carbon` | mobility demand + charging market + emissions |
| `logistics_carbon_credit` | transport dispatch + carbon credits + cost competition |
| `edge_computing_energy` | compute + bandwidth + energy + carbon cost |

The electricity-carbon-green scenario is the fully implemented baseline. Other
scenarios provide runnable templates and shared metadata contracts for extension.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --describe
python main.py --quick --output-dir results_quick
python main.py --episodes 1000 --output-dir results_1000
```

Use CUDA by installing a CUDA-enabled PyTorch build. To select a specific GPU:

```bash
set CUDA_VISIBLE_DEVICES=1
python main.py --episodes 1000 --output-dir results_1000_rtx6000
```

## Agent Roles

The core package defines general roles:

- `ProducerAgent`
- `ConsumerAgent`
- `StorageAgent`
- `AggregatorAgent`
- `TraderAgent`
- `RegulatorAgent`
- `MarketOperator`

Scenario-specific agents are composed from these roles. For example:

- `ThermalGenAgent = ProducerAgent + carbon_emission`
- `RenewableAgent = ProducerAgent + certificate_generation`
- `CloudProviderAgent = ProducerAgent + compute_resource + energy_cost`
- `EVFleetAgent = StorageAgent + mobility_demand`
- `LogisticsCompanyAgent = ProducerAgent/ConsumerAgent + transport_emission`

## Research Positioning

CoupledMarket-MARLBench studies how multiple agents learn strategies across
interdependent markets while accounting for resources, policy, externalities,
constraints, and settlement rules. This makes the repository useful beyond the
energy-market demo and gives a stable foundation for broader coupled-market MARL
experiments.
