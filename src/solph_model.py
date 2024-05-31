from oemof import solph
import pandas as pd
import numpy as np


def build_and_run_dispatch_optimization(date_time_index, time_series_agg):

    es = solph.EnergySystem(
        timeindex=date_time_index, infer_last_interval=False
    )

    b_gas = solph.Bus("gas bus")
    b_electricity = solph.Bus("electricity bus")
    b_heat = solph.Bus("heat bus")

    source_gas = solph.components.Source(
        "gas grid import",
        outputs={b_gas: solph.Flow(variable_costs=time_series_agg["gas_price"] + time_series_agg["co2_price"])}
    )
    source_electricity = solph.components.Source(
        "electricity grid import",
        outputs={b_electricity: solph.Flow(variable_costs=time_series_agg["el_spot_price"])}
    )
    sink_heat = solph.components.Sink(
        "heat demand",
        inputs={b_heat: solph.Flow(fix=time_series_agg["heat"], nominal_value=1)}
    )

    heat_pump = solph.components.Converter(
        label="heat pump",
        inputs={b_electricity: solph.Flow()},
        outputs={b_heat: solph.Flow(nominal_value=100)},
        conversion_factors={b_heat: 3.5}
    )
    boiler = solph.components.Converter(
        label="gas boiler",
        inputs={b_gas: solph.Flow()},
        outputs={b_heat: solph.Flow(nominal_value=100)},
        conversion_factors={b_heat: 0.9}
    )

    heat_slack = solph.components.Source(
        label="heat slack",
        outputs={b_heat: solph.Flow(variable_costs=1000)}
    )

    storage = solph.components.GenericStorage(
        label="heat storage",
        inputs={b_heat: solph.Flow(nominal_value=50)},
        outputs={b_heat: solph.Flow(nominal_value=50)},
        nominal_storage_capacity=24 * 50,
        initial_storage_level=0.5,
        balanced=True
    )

    es.add(
        b_gas, b_electricity, b_heat,
        source_electricity, source_gas, sink_heat, heat_slack,
        heat_pump, boiler, storage
    )

    model = solph.Model(es)

    _ = model.solve("gurobi")

    results = solph.views.convert_keys_to_strings(model.results())

    return results


def generate_date_time_index(clustering):
    hours = [0] + np.cumsum(clustering.index.get_level_values(level="duration").values).tolist()

    start_time = pd.Timestamp('2023-01-01 00:00:00')
    time_deltas = pd.to_timedelta(hours, unit='h')

    date_times = start_time + time_deltas
    date_time_index = pd.DatetimeIndex(date_times)
    return date_time_index
