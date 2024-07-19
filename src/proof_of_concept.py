import matplotlib.pyplot as plt
import pandas as pd

from es_opt import (calc_key_params, fill_year, optimization, postprocessing,
                    preprocessing, run_optimization_timed)

# MARK: Base Case
ts_in = pd.read_csv("time_series.csv", index_col=0, parse_dates=True)
ts_in = ts_in.drop(columns="ef_om")

results = optimization(agg_data=ts_in)
unitdata_base = postprocessing(results)
opex_total_base, heat_prod_shares_base, heat_prod_total_base = calc_key_params(
    unitdata_base, ts_in
)

# MARK: Aggregation
noTypicalPeriods = 10
hoursPerPeriod = 168

ts_in = pd.read_csv("time_series.csv", index_col=0, parse_dates=True)
ts_in = ts_in.drop(columns="ef_om")
missing_hours = ts_in.shape[0] % hoursPerPeriod
if missing_hours > 0:
    ts_in = ts_in.iloc[:-missing_hours, :]
print(ts_in.shape)

aggregation, agg_data = preprocessing(
    noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
    data=ts_in
)
print(agg_data)

results = optimization(agg_data=agg_data)
print(results)

desagg_data = postprocessing(
    results=results, agg_data=agg_data, aggregation=aggregation
)
print(desagg_data)

# MARK: Plots
# Comparison heat production aggregated vs. real
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(
    desagg_data['heat bus to heat demand'],
    label='Desaggregierte Wärmeproduktion'
)
ax.plot(ts_in['heat'], label='Gemessene Wärmeproduktion')

ax.legend()
ax.grid()
ax.set_ylabel('Stündliche Wärmeproduktion in MW')

# heat storage storage content
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(desagg_data['heat storage storage content'])

ax.grid()
ax.set_ylabel('Wärmespeicherfüllstand in MWh')

produnits = [
    'heat pump to heat bus',
    'gas boiler to heat bus',
    'heat slack to heat bus'
]
aggdiff = desagg_data[produnits] - unitdata_base[produnits]
print(aggdiff.describe())

fig, ax = plt.subplots(figsize=(16, 5))

ax.bar(
    list(range(1, len(desagg_data)+1)),
    desagg_data['heat pump to heat bus'],
    label='Einsatz desaggregiert'
)
ax.bar(
    list(range(1, len(aggdiff)+1)),
    aggdiff['heat pump to heat bus'],
    label='Abweichung Basecase'
)

ax.legend()
ax.grid(axis='y')
ax.set_ylabel('Abweichung Wärmeproduktion Wärmepumpe in MW')

fig, ax = plt.subplots(figsize=(16, 5))

ax.bar(
    list(range(1, len(desagg_data)+1)),
    desagg_data['gas boiler to heat bus'],
    label='Einsatz desaggregiert'
)
ax.bar(
    list(range(1, len(aggdiff)+1)),
    aggdiff['gas boiler to heat bus'],
    label='Abweichung Basecase'
)

ax.legend()
ax.grid(axis='y')
ax.set_ylabel('Abweichung Wärmeproduktion Gaskessel in MW')

fig, ax = plt.subplots(figsize=(16, 5))

ax.bar(
    list(range(1, len(desagg_data)+1)),
    desagg_data['heat slack to heat bus'],
    label='Einsatz desaggregiert'
)
ax.bar(
    list(range(1, len(aggdiff)+1)),
    aggdiff['heat slack to heat bus'],
    label='Abweichung Basecase'
)

ax.legend()
ax.grid(axis='y')
ax.set_ylabel('Abweichung Wärmeproduktion Wärmequelle in MW')

plt.show()
