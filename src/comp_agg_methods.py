import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from es_opt import (calc_key_params, fill_year, optimization, postprocessing,
                    preprocessing, run_optimization_timed)

# MARK: Base Case
ts_in = pd.read_csv("src\\time_series.csv", index_col=0, parse_dates=True)
ts_in = ts_in.drop(columns="ef_om")

results = optimization(agg_data=ts_in)
unitdata_base = postprocessing(results)
opex_total_base, heat_prod_shares_base, heat_prod_total_base = calc_key_params(
    unitdata_base, ts_in
)

# MARK: Aggregation
noTypicalPeriods = 10
hoursPerPeriod = 120

kwargses = {
    'K means w/o extreme periods': {
        'clusterMethod': 'k_means'
    },
    'K means w/ extreme periods': {
        'clusterMethod': 'k_means',
        'extremePeriodMethod': 'new_cluster_center',
        'addPeakMin': ['heat'],
        'addPeakMax': ['heat']
    },
    'Hierarchical w/o extreme periods': {
        'clusterMethod': 'hierarchical'
    },
    'Hierarchical w/ extreme periods': {
        'clusterMethod': 'hierarchical',
        'extremePeriodMethod': 'new_cluster_center',
        'addPeakMin': ['heat'],
        'addPeakMax': ['heat']
    }
}

fig, axs = plt.subplots(2, 2, sharex='row', sharey='col', figsize=(13, 13))

heatcol = 'heat bus to heat demand'

fig.suptitle(f'{noTypicalPeriods = }, {hoursPerPeriod = }')

for i, (name, kwargs) in enumerate(kwargses.items()):
    aggregation, agg_data = preprocessing(
        noTypicalPeriods=noTypicalPeriods, hoursPerPeriod=hoursPerPeriod,
        data=ts_in, **kwargs
    )
    results = optimization(agg_data=agg_data)
    desagg_data = postprocessing(
        results=results, agg_data=agg_data, aggregation=aggregation
    )

    corr = np.corrcoef(
        unitdata_base[heatcol].to_numpy(), desagg_data[heatcol].to_numpy()
    )

    row = int(i % 2)
    col = int(np.floor(i / 2))

    axs[row, col].plot(
        (0, unitdata_base[heatcol].max()*1.05),
        (0, unitdata_base[heatcol].max()*1.05),
        color='k'
    )
    axs[row, col].scatter(unitdata_base[heatcol], desagg_data[heatcol])

    axs[row, col].set_axisbelow(True)
    axs[row, col].grid()
    axs[row, col].set_xlim(left=0, right=unitdata_base[heatcol].max()*1.05)
    axs[row, col].set_ylim(bottom=0, top=unitdata_base[heatcol].max()*1.05)
    axs[row, col].set_xlabel('Real heat load in MWh')
    axs[row, col].set_ylabel('Aggregated heat load in MWh')
    axs[row, col].set_title(name)

    axs[row, col].annotate(
        f'R = {corr[0, 1]:.4f}',
        (0.02, 0.95), xycoords='axes fraction',
        bbox=dict(boxstyle='round', fc='#f1f1f1')
    )

plt.show()
