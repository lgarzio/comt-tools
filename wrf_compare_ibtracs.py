#!/usr/bin/env python

"""
Author: Lori Garzio on 1/7/2021
Last modified: 1/7/2021
Compares SLP from IBTrACS data to different WRF model runs for Hurricane Irene
"""

import numpy as np
import os
import glob
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import functions.common as cf


def format_date_axis(axis):
    datef = mdates.DateFormatter('%m-%d %H')
    axis.xaxis.set_major_formatter(datef)


def main(ddir):
    save_dir = os.path.join(os.path.dirname(ddir), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    wrf_files = sorted(glob.glob(os.path.join(ddir, '*subset.nc')))
    wrf_dict = {}
    for wf in wrf_files:
        fname = 'wrf' + wf.split('/')[-1].split('_')[-2][-1]
        ncfile = xr.open_dataset(wf, mask_and_scale=False)
        if fname == 'wrf1':
            lon_wrf = ncfile.XLONG
            lat_wrf = ncfile.XLAT
            wrf_time = ncfile.Time
            # add time to dictionary
            wrf_dict['tm'] = pd.to_datetime(wrf_time.values)

            # get data from IBTrACS file. Hurricane Irene 2011
            ibtracs = xr.open_dataset(os.path.join(ddir, 'IBTrACS.ALL.v04r00.nc'))
            irene_ib = ibtracs.sel(storm=12484)
            ib_slp = irene_ib.usa_pres

            # interpolate IBTrACS data to wrf timestamps
            ib_slp = ib_slp.swap_dims({'date_time': 'time'})
            ib_slp = ib_slp[~np.isnan(ib_slp)]  # remove nans
            ib_slp_interp = ib_slp.interp(time=wrf_time)
            ib_slp_interp_stdev = ib_slp_interp.std().values.item()

        # calculate the minimum SLP for each time stamp
        slp = ncfile.SLP
        minslp = slp.min(dim=('south_north', 'west_east'))

        # find the lat/lon for each minimum for each time stamp
        minslp_idx = slp.argmin(dim=('south_north', 'west_east'))
        lons, lats = lon_wrf[minslp_idx], lat_wrf[minslp_idx]

        # add data to dictionary
        wrf_dict[fname] = {}
        wrf_dict[fname]['minslp'] = {}
        wrf_dict[fname]['minslp']['values'] = minslp.values
        wrf_dict[fname]['minslp']['lons'] = lons.values
        wrf_dict[fname]['minslp']['lats'] = lats.values

        # calculate the correlation between model and obs (IBTrACS) and add to dictionary
        wrf_dict[fname]['minslp']['corr'] = cf.pearson_correlation(ib_slp_interp, minslp)

        # calculate the standard deviation of model and add to dictionary
        wrf_dict[fname]['minslp']['std'] = minslp.std().values.item()

        # calculate CRMSE between model and obs (IBTrACS) and add to dictionary
        wrf_dict[fname]['minslp']['crmse'] = cf.crmse(ib_slp_interp, minslp)

        # calculate bias between model and obs (IBTrACS) and add to dictionary
        wrf_dict[fname]['minslp']['bias'] = cf.model_bias(ib_slp_interp, minslp)

    # define colors and labels for plots
    colors = ['tab:blue', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:purple']
    labels = ['k-kl', r'k-$\epsilon$', r'k-$\omega$', 'HyCOM', 'MUR SST']

    # plot map of minimum SLP
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection=proj))
    ax.plot(ib_slp_interp.lon, ib_slp_interp.lat, transform=ccrs.PlateCarree(), color='black', label='NHC best track')

    plt_vars = ['minslp']
    for pv in plt_vars:
        cnt = 0
        for key, item in wrf_dict.items():
            if 'wrf' in key:
                ax.plot(item[pv]['lons'], item[pv]['lats'], transform=proj, color=colors[cnt], label=labels[cnt])
                cnt += 1

    # add map features
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax.set_extent([-78, -73, 34, 42])
    ax.legend(fontsize=10)
    gl.top_labels = gl.right_labels = False
    gl.rotate_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'irene_track.png'), format='png', dpi=300)

    # plot timeseries of minimum SLP
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ib_slp_interp.time.values, ib_slp_interp.values, label='NHC best track', color='black', lw=2)

    for pv in plt_vars:
        cnt = 0
        for key, item in wrf_dict.items():
            if 'wrf' in key:
                ax.plot(wrf_dict['tm'], item[pv]['values'], color=colors[cnt], label=labels[cnt], lw=2)
                cnt += 1

    ax.set_ylim([950, 980])
    plt.xlabel('Date (mm-dd HH)')
    plt.ylabel('Pressure (millibars)')
    plt.title('Hurricane Irene Minimum Pressure')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    format_date_axis(ax)
    plt.xticks(rotation=30)

    plt.savefig(os.path.join(save_dir, 'minslp_timeseries.png'), format='png', dpi=300, bbox_inches='tight')

    # Taylor diagram of WRF model runs, using the IBTrACS data as the observations
    angle_lim = np.pi / 2
    std_lim = 1.75

    fig, ax = cf.taylor_template(angle_lim, std_lim)

    cnt = 0
    for key, item in wrf_dict.items():
        if 'wrf' in key:
            theta = np.arccos(item['minslp']['corr'])
            rr = item['minslp']['std'] / ib_slp_interp_stdev
            ax.plot(theta, rr, 's', color=colors[cnt], markersize=8)
            cnt += 1

    ax.plot(0, 1, 'o', label='Obs', markersize=8)

    rs, ts = np.meshgrid(np.linspace(0, std_lim), np.linspace(0, angle_lim))
    rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))

    contours = ax.contour(ts, rs, rms, 3, colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)
    plt.savefig(os.path.join(save_dir, 'minslp_taylor_normalized.png'), format='png', dpi=300)


if __name__ == '__main__':
    data_dir = '/Users/lgarzio/Documents/rucool/Miles/NOAA_COMT/data'  # location of data files
    main(data_dir)
