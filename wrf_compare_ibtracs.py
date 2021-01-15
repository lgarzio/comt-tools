#!/usr/bin/env python

"""
Author: Lori Garzio on 1/7/2021
Last modified: 1/15/2021
Compares SLP from IBTrACS data to different WRF model runs for Hurricane Irene. Interpolated WRF time to IBTrACS time
"""

import numpy as np
import os
import glob
import xarray as xr
import pickle
import pandas as pd
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import functions.common as cf


def format_date_axis(axis):
    datef = mdates.DateFormatter('%m-%d %H')
    axis.xaxis.set_major_formatter(datef)
    plt.xticks(rotation=30)


def plot_map_tracks(obs_slp, mdict, plotting_variables, plotting_labels, save_figname):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection=proj))
    ax.plot(obs_slp.lon, obs_slp.lat, transform=ccrs.PlateCarree(), color='black', label='NHC best track')

    for pv in plotting_variables:
        cnt = 0
        for key, item in mdict.items():
            if 'wrf' in key:
                ax.plot(item[pv]['lons'], item[pv]['lats'], transform=proj, color=plotting_labels[key][1],
                        label=plotting_labels[key][0])
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
    plt.savefig(save_figname, format='png', dpi=300)


def plot_timeseries(obs_slp, mdict, plotting_variables, pltdata, plotting_labels, xvar, ylabel, sfig, title=None, ylims=None,
                    best_track=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    if best_track:
        ax.plot(obs_slp.time.values, obs_slp.values, label='NHC best track', color='black', lw=2)

    for pv in plotting_variables:
        cnt = 0
        for key, item in mdict.items():
            if 'wrf' in key:
                ax.plot(item[pv]['xvar'], item[pv][pltdata], color=plotting_labels[key][1],
                        label=plotting_labels[key][0], lw=2)
                cnt += 1

    if ylims:
        ax.set_ylim([950, 980])
    plt.xlabel('Date (mm-dd HH)')
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    format_date_axis(ax)

    plt.savefig(sfig, format='png', dpi=300, bbox_inches='tight')


def plot_timeseries_trackerror(obs_slp, mdict, plotting_variables, pltdata, plotting_labels, ylabel, sfig,
                               title=None, ylims=None, best_track=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    if best_track:
        ax.plot(obs_slp.time.values, obs_slp.values, label='NHC best track', color='black', lw=2)

    for pv in plotting_variables:
        cnt = 0
        for key, item in mdict.items():
            if 'wrf' in key:
                ax.plot(obs_slp.time.values, item[pv][pltdata], color=plotting_labels[key][1],
                        label=plotting_labels[key][0], lw=2)
                cnt += 1

    if ylims:
        ax.set_ylim([950, 980])
    plt.xlabel('Date (mm-dd HH)')
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    format_date_axis(ax)

    plt.savefig(sfig, format='png', dpi=300, bbox_inches='tight')


def main(ddir):
    save_dir = os.path.join(os.path.dirname(ddir), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # get the plotting colors and labels: [label, color]
    with open('plt_labels.pickle', 'rb') as handle:
        plt_labs = pickle.load(handle)

    wrf_files = sorted(glob.glob(os.path.join(ddir, '*subset.nc')))
    wrf_dict = {}
    for wf in wrf_files:
        fname = 'wrf' + wf.split('/')[-1].split('_')[-2][-1]
        ncfile = xr.open_dataset(wf, mask_and_scale=False)
        if fname == 'wrf1':
            # get WRF coordinates
            lon_wrf = ncfile.XLONG
            lat_wrf = ncfile.XLAT

            # get data from IBTrACS file. Hurricane Irene 2011
            ibtracs = xr.open_dataset(os.path.join(ddir, 'IBTrACS.ALL.v04r00.nc'))
            irene_ib = ibtracs.sel(storm=12484)
            irene_ib = irene_ib.swap_dims({'date_time': 'time'})
            ib_slp = irene_ib.usa_pres
            ib_slp = ib_slp[~np.isnan(ib_slp)]  # remove nans
            #ib_slp_interp = ib_slp.interp(time=wrf_time)
            #ib_slp_interp_stdev = ib_slp_interp.std().values.item()

        # calculate the minimum SLP for each time stamp
        slp = ncfile.SLP
        minslp = slp.min(dim=('south_north', 'west_east'))

        # find the lat/lon for each minimum for each time stamp
        minslp_idx = slp.argmin(dim=('south_north', 'west_east'))
        lons, lats = lon_wrf[minslp_idx], lat_wrf[minslp_idx]

        # interpolate WRF data to IBTrACS time
        minslp_interp = minslp.interp(Time=ib_slp.time)
        lons_interp = lons.interp(Time=ib_slp.time)
        lats_interp = lats.interp(Time=ib_slp.time)

        # remove NaNs (the IBTrACS data encompass a longer time range)
        nanidx = ~np.isnan(minslp_interp)
        minslp_interp = minslp_interp[nanidx]
        lons_interp = lons_interp[nanidx]
        lats_interp = lats_interp[nanidx]
        ib_slp = ib_slp[nanidx]
        ib_slp_stdev = ib_slp.std().values.item()

        # add data to dictionary
        wrf_dict[fname] = {}
        wrf_dict[fname]['minslp'] = {}
        wrf_dict[fname]['minslp']['wrf_time'] = minslp.Time.values
        wrf_dict[fname]['minslp']['values'] = minslp.values
        wrf_dict[fname]['minslp']['lons'] = lons
        wrf_dict[fname]['minslp']['lats'] = lats
        wrf_dict[fname]['minslp']['values_interp'] = minslp_interp.values
        wrf_dict[fname]['minslp']['lons_interp'] = lons_interp.values
        wrf_dict[fname]['minslp']['lats_interp'] = lats_interp.values

        # calculate track error geodesic distance and angle (degrees clockwise from north)
        geod = Geodesic.WGS84
        diff_loc = np.array([])
        trerror_angle = np.array([])
        for idx, iblat in enumerate(ib_slp.lat.values):
            g = geod.Inverse(iblat, ib_slp.lon.values[idx], lats_interp.values[idx], lons_interp.values[idx])
            diff_loc = np.append(diff_loc, g['s12'] * .001)
            trerror_angle = np.append(trerror_angle, np.round(g['azi1'], 2))
        wrf_dict[fname]['minslp']['track_error_km'] = diff_loc
        wrf_dict[fname]['minslp']['track_error_angle'] = trerror_angle

        # calculate the correlation between model and obs (IBTrACS) and add to dictionary
        wrf_dict[fname]['minslp']['corr'] = cf.pearson_correlation(ib_slp, minslp_interp)

        # calculate the standard deviation of model and add to dictionary
        wrf_dict[fname]['minslp']['std'] = minslp_interp.std().values.item()

        # calculate CRMSE between model and obs (IBTrACS) and add to dictionary
        wrf_dict[fname]['minslp']['crmse'] = cf.crmse(ib_slp, minslp_interp)

        # calculate bias between model and obs (IBTrACS) and add to dictionary
        wrf_dict[fname]['minslp']['bias'] = cf.model_bias(ib_slp, minslp_interp)

    # print track error and angle
    df = pd.DataFrame()
    df['ibtracs_time'] = ib_slp.time.values
    df['ibtracs_lat'] = np.round(ib_slp.lat.values, 1)
    df['ibtracs_lon'] = np.round(ib_slp.lon.values, 1)
    df['wrf1_track_error_km'] = np.round(wrf_dict['wrf1']['minslp']['track_error_km'], 2)
    df['wrf1_track_error_angle'] = wrf_dict['wrf1']['minslp']['track_error_angle']
    df['wrf3_track_error_km'] = np.round(wrf_dict['wrf3']['minslp']['track_error_km'], 2)
    df['wrf3_track_error_angle'] = wrf_dict['wrf3']['minslp']['track_error_angle']
    df['wrf4_track_error_km'] = np.round(wrf_dict['wrf4']['minslp']['track_error_km'], 2)
    df['wrf4_track_error_angle'] = wrf_dict['wrf4']['minslp']['track_error_angle']
    df['wrf5_track_error_km'] = np.round(wrf_dict['wrf5']['minslp']['track_error_km'], 2)
    df['wrf5_track_error_angle'] = wrf_dict['wrf5']['minslp']['track_error_angle']
    df['wrf6_track_error_km'] = np.round(wrf_dict['wrf6']['minslp']['track_error_km'], 2)
    df['wrf6_track_error_angle'] = wrf_dict['wrf6']['minslp']['track_error_angle']

    df.to_csv(os.path.join(save_dir, 'track_error.csv'), index=False)

    # define variables to plot
    plt_vars = ['minslp']

    # plot map of minimum SLP (plot raw WRF data, not interpolated)
    sfig = os.path.join(save_dir, 'irene_track.png')
    plot_map_tracks(ib_slp, wrf_dict, plt_vars, plt_labs, sfig)

    # plot timeseries of minimum SLP (plot raw WRF data, not interpolated)
    time_var = 'wrf_time'
    ylab = 'Pressure (millibars)'
    title = 'Hurricane Irene Minimum Pressure'
    ylim = [950, 980]
    sfig = os.path.join(save_dir, 'minslp_timeseries.png')
    plot_timeseries(ib_slp, wrf_dict, plt_vars, 'values', plt_labs, time_var, ylab, sfig, title, ylim, best_track='yes')

    # plot timeseries of track error
    ylab = 'Track Error (km)'
    sfig = os.path.join(save_dir, 'track_error_timeseries.png')
    plot_timeseries_trackerror(ib_slp, wrf_dict, plt_vars, 'track_error_km', plt_labs, ylab, sfig)

    # plot track error by latitude
    fig, ax = plt.subplots(figsize=(6, 8))
    for pv in plt_vars:
        cnt = 0
        for key, item in wrf_dict.items():
            if 'wrf' in key:
                ax.plot(item[pv]['track_error_km'], item[pv]['lats_interp'], color=plt_labs[key][1],
                        label=plt_labs[key][0], lw=2)
                cnt += 1
    plt.xlabel('Track Error (km)')
    plt.ylabel('Interpolated Latitude')
    #plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'track_error_latitude.png'), format='png', dpi=300, bbox_inches='tight')

    # Taylor diagram of WRF model runs, using the IBTrACS data as the observations
    angle_lim = np.pi / 2
    std_lim = 1.75

    fig, ax = cf.taylor_template(angle_lim, std_lim, 111)  # 111 = 1 plot

    cnt = 0
    for key, item in wrf_dict.items():
        if 'wrf' in key:
            theta = np.arccos(item['minslp']['corr'])
            rr = item['minslp']['std'] / ib_slp_stdev
            ax.plot(theta, rr, 's', color=plt_labs[key][1], markersize=8)
            cnt += 1

    ax.plot(0, 1, 'o', color='blue', label='Obs', markersize=8)

    rs, ts = np.meshgrid(np.linspace(0, std_lim), np.linspace(0, angle_lim))
    rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))

    contours = ax.contour(ts, rs, rms, 3, colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)
    plt.savefig(os.path.join(save_dir, 'minslp_taylor_normalized.png'), format='png', dpi=300)


if __name__ == '__main__':
    data_dir = '/Users/lgarzio/Documents/rucool/Miles/NOAA_COMT/data'  # location of data files
    main(data_dir)
