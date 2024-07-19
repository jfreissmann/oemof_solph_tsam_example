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
hoursPerPeriod = 120

aggregation, agg_data = preprocessing(
    noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
    data=ts_in
)
results = optimization(agg_data=agg_data)
desagg_data = postprocessing(
    results=results, agg_data=agg_data, aggregation=aggregation
)

aggregation_extreme, agg_data_extreme = preprocessing(
    noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
    data=ts_in, extremePeriodMethod='new_cluster_center'
)
results_extreme = optimization(agg_data=agg_data_extreme)
desagg_data_extreme = postprocessing(
    results=results_extreme, agg_data=agg_data_extreme,
    aggregation=aggregation_extreme
)

fig, axs = plt.subplots(1, 2, sharey='col', figsize=(14, 6))

heatcol = 'heat bus to heat demand'

axs[0].scatter(unitdata_base[heatcol], desagg_data[heatcol])
axs[1].scatter(unitdata_base[heatcol], desagg_data_extreme[heatcol])

for ax in axs:
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_xlim(left=0, right=unitdata_base[heatcol].max()*1.05)
    ax.set_ylim(bottom=0, top=unitdata_base[heatcol].max()*1.05)
    ax.set_xlabel('Real heat load in MWh')
    ax.set_ylabel('Aggregated heat load in MWh')

plt.show()
