"""Logistics dispatch and carbon-credit competition scenario template."""

from market_env.scenarios.template import TemplateCoupledMarketEnv


def make_env(**kwargs):
    return TemplateCoupledMarketEnv(
        scenario_name="logistics_carbon_credit",
        markets=["transport_market", "fuel_market", "carbon_credit_market"],
        resources=["transport_capacity", "fuel", "carbon_credit"],
        agent_roles=["logistics_company", "shipper", "credit_trader"],
        **kwargs,
    )
