from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tsam.timeseriesaggregation as tsam
from oemof import solph


def preprocessing(noTypicalPeriods, hoursPerPeriod, data,
                  clusterMethod='k_means', **kwargs):
    aggregation = tsam.TimeSeriesAggregation(
        data,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod='k_means',
        **kwargs
    )

    agg_data = aggregation.createTypicalPeriods()

    agg_data.index.names = ['TypicalPeriod', 'TimeStep']
    agg_data.reset_index(inplace=True)

    return aggregation, agg_data

def optimization(agg_data):
    timeindex = pd.date_range('2019-01-01 00:00', freq='h', periods=agg_data.shape[0])
    es = solph.EnergySystem(
        timeindex=timeindex, infer_last_interval=True
    )

    b_gas = solph.Bus("gas bus")
    b_electricity = solph.Bus("electricity bus")
    b_heat = solph.Bus("heat bus")

    source_gas = solph.components.Source(
        "gas source",
        outputs={b_gas: solph.Flow(variable_costs=agg_data["gas_price"] + agg_data["co2_price"])}
    )
    source_electricity = solph.components.Source(
        "electricity source",
        outputs={b_electricity: solph.Flow(variable_costs=agg_data["el_spot_price"])}
    )
    sink_heat = solph.components.Sink(
        "heat demand",
        inputs={b_heat: solph.Flow(fix=agg_data["heat"], nominal_value=1)}
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

def postprocessing(results, agg_data=None, aggregation=None):
    unitdata = pd.DataFrame()
    for vertex, data in results.items():
        if vertex[-1] != 'None':
            unitdata[' to '.join(vertex)] = data['sequences']['flow']
        else:
            unitdata[f'{vertex[0]} storage content'] = data['sequences']['storage_content']

    if agg_data is not None and aggregation is not None:
        unitdata = unitdata.dropna()
        unitdata = unitdata.reset_index(drop=True)

        unitdata = pd.concat([unitdata, agg_data[['TypicalPeriod', 'TimeStep']]], axis=1)
        unitdata = unitdata.set_index(['TypicalPeriod', 'TimeStep'])

        matched_indices = aggregation.indexMatching()

        periods = unitdata.index.get_level_values('TypicalPeriod')
        timesteps = unitdata.index.get_level_values('TimeStep')

        unitdata_flat = pd.DataFrame(unitdata.values, columns=unitdata.columns)
        unitdata_flat['PeriodNum'] = periods
        unitdata_flat['TimeStep'] = timesteps

        desagg_data = matched_indices.reset_index().merge(
            unitdata_flat,
            how='left',
            left_on=['PeriodNum', 'TimeStep'],
            right_on=['PeriodNum', 'TimeStep']
        ).set_index(matched_indices.index)

        desagg_data = desagg_data.drop(columns=['Date', 'PeriodNum', 'TimeStep'])

        return desagg_data

    else:
        return unitdata.dropna()

def fill_year(desagg_data, aggregation):
    matched_indices = aggregation.indexMatching()

    if desagg_data.index[0].is_leap_year:
        nr_missing = 8784 - len(desagg_data)
    else:
        nr_missing = 8760 - len(desagg_data)

    lastperiod = matched_indices[matched_indices['PeriodNum'] == matched_indices['PeriodNum'].iloc[-1]]
    lastperiod.drop_duplicates(inplace=True)
    lpdata = desagg_data.loc[lastperiod.iloc[:nr_missing].index , :]

    print(lpdata.shape)

    missing_index = pd.date_range(
        start=f'{desagg_data.index[-1]+pd.Timedelta("1h")}',
        end=f'{desagg_data.index[-1].year}-12-31 23:00',
        freq='h'
    )
    missing_data = pd.DataFrame(index=missing_index, data=lpdata.values, columns=lpdata.columns)

    return pd.concat([desagg_data, missing_data], axis=0)

def calc_key_params(desagg_data, ts_in):
    el_cost = (desagg_data['electricity source to electricity bus'].values * ts_in['el_spot_price'].values).sum()
    gas_cost = (desagg_data['gas source to gas bus'].values * (
        ts_in['gas_price'].values + ts_in['co2_price'].values
    )).sum()
    slack_cost = (desagg_data['heat slack to heat bus'].values * 1000).sum()

    opex_total = el_cost + gas_cost  # + slack_cost

    heat_prod_shares = (
        desagg_data[['heat pump to heat bus', 'gas boiler to heat bus', 'heat slack to heat bus']].sum()
        / desagg_data[['heat pump to heat bus', 'gas boiler to heat bus', 'heat slack to heat bus']].sum().sum()
        )
    heat_prod_shares

    heat_prod_total = desagg_data['heat bus to heat demand'].sum()

    return opex_total, heat_prod_shares, heat_prod_total

def run_optimization_timed(noTypicalPeriods, hoursPerPeriod):
    ts_in = pd.read_csv("time_series.csv", index_col=0, parse_dates=True)
    ts_in = ts_in.drop(columns="ef_om")

    starttime = time()
    aggregation, agg_data = preprocessing(noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod, data=ts_in)
    aggregation_time = time() - starttime

    starttime = time()
    results = optimization(agg_data=agg_data)
    optimization_time = time() - starttime

    starttime = time()
    desagg_data = postprocessing(results=results, agg_data=agg_data, aggregation=aggregation)

    if desagg_data.index[0].is_leap_year:
        target_hours = 8784
        missing_hours = target_hours - len(desagg_data)
    else:
        target_hours = 8760
        missing_hours = target_hours - len(desagg_data)

    if missing_hours > 0:  # Is this case even occurring?
        desagg_data = fill_year(desagg_data=desagg_data, aggregation=aggregation)
    elif missing_hours < 0:
        desagg_data = desagg_data.iloc[:target_hours, :]

    disaggregation_time = time() - starttime

    opex_total, heat_prod_shares, heat_prod_total = calc_key_params(desagg_data=desagg_data, ts_in=ts_in)

    return aggregation_time, optimization_time, disaggregation_time, desagg_data, opex_total, heat_prod_shares, heat_prod_total, desagg_data
