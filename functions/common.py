#! /usr/bin/env python

"""
Author: Lori Garzio on 1/7/2021
Last modified: 1/7/2021
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def crmse(data1, data2):
    res = np.sqrt(np.nansum((((data1 - data1.mean()) - (data2 - data2.mean())) ** 2) / (len(data2) - 1)))
    return res


def model_bias(obs_data, model_data):
    b = (obs_data.mean() - model_data.mean()).values.item()
    return b


def pearson_correlation(data1, data2):
    # input data must be xarray.Datasets
    pc = xr.corr(data1, data2).values.item()
    return pc


def taylor_template(angle_lim, std_lim):
    # written by Maria Aristizabal: https://github.com/MariaAristizabal/Evaluation_surf_metrics_Dorian_figures
    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from matplotlib.projections import PolarAxes
    from mpl_toolkits.axisartist.grid_finder import (FixedLocator,
                                                     DictFormatter)

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()

    min_corr = np.round(np.cos(angle_lim), 1)
    CCgrid = np.concatenate((np.arange(min_corr * 10, 10, 2.0) / 10., [0.9, 0.95, 0.99]))
    CCpolar = np.arccos(CCgrid)
    gf = FixedLocator(CCpolar)
    tf = DictFormatter(dict(zip(CCpolar, map(str, CCgrid))))

    STDgrid = np.arange(0, std_lim, .5)
    gfs = FixedLocator(STDgrid)
    tfs = DictFormatter(dict(zip(STDgrid, map(str, STDgrid))))

    ra0, ra1 = 0, angle_lim
    cz0, cz1 = 0, std_lim
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=gf,
        tick_formatter1=tf,
        grid_locator2=gfs,
        tick_formatter2=tfs)

    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text("Correlation")
    ax1.axis['top'].label.set_size(14)

    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["left"].label.set_text("Normalized Standard Deviation")
    ax1.axis['left'].label.set_size(14)

    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].major_ticklabels.set_axis_direction("left")

    ax1.axis["bottom"].set_visible(False)
    ax1 = ax1.get_aux_axes(tr)

    plt.grid(linestyle=':', alpha=0.5)

    return fig, ax1


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    :param u: west/east direction (wind from the west is positive, from the east is negative)
    :param v: south/noth direction (wind from the south is positive, from the north is negative)
    :returns WSPD: wind speed calculated from the u and v wind components
    """
    WSPD = np.sqrt(np.square(u)+np.square(v))

    return WSPD
