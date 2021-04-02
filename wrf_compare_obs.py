#!/usr/bin/env python

"""
Author: Lori Garzio on 1/11/2021
Last modified: 4/1/2021
Compares SST, Air Temperature, SLP and Wind Speed from NDBC buoys 44009 and 44065, data from the Tuckerton met tower,
and data from RU-16 to different WRF and ROMS model runs for Hurricane Irene. Interpolated observation time to WRF time.
Adjusted buoy wind speeds to 10m using a wind power law.
"""

import numpy as np
import os
import glob
from erddapy import ERDDAP
import datetime as dt
import xarray as xr
import pickle
import pandas as pd
import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import functions.common as cf
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def get_erddap_dataset(server, protocol, file_type, ds_id, var_list=None):
    e = ERDDAP(server=server,
               protocol=protocol,
               response=file_type)
    e.dataset_id = ds_id
    if var_list:
        e.variables = var_list
    ds = e.to_xarray()
    ds = ds.sortby(ds.time)
    return ds


def append_obs(data_dict, obs_key, varname, tm, values, interp_value, interp_stdev):
    try:
        data_dict[obs_key][varname]
    except KeyError:
        data_dict[obs_key][varname] = {}
        data_dict[obs_key][varname]['obs_tm'] = tm
        data_dict[obs_key][varname]['obs_values'] = values
        data_dict[obs_key][varname]['obs_values_interp'] = interp_value
        data_dict[obs_key][varname]['obs_interp_std'] = interp_stdev


def find_obs_loc(wrf_fname, obs_loc):
    # find the lat/lon index in the dataset that corresponds with the observation lat/lon
    ds = xr.open_dataset(wrf_fname, mask_and_scale=False)

    a = abs(ds.XLAT[:, :, 1] - obs_loc[0]) + abs(ds.XLONG[:, :, 1] - obs_loc[1])
    ia, ja = np.unravel_index(a.argmin(), a.shape)

    return ia, ja


def find_obs_loc_roms(roms_fname, obs_loc):
    # find the lat/lon index in the dataset that corresponds with the observation lat/lon
    ds = xr.open_dataset(roms_fname, mask_and_scale=False)

    a = abs(ds.lat_rho-obs_loc[0]) + abs(ds.lon_rho - obs_loc[1])
    ia, ja = np.unravel_index(a.argmin(), a.shape)

    return ia, ja


def get_buoy_data(bfile, buoy, plot_var, vname, interp_time, data_dict):
    anemometer_height = {'44009': 4.9, '44065': 4.1}
    ds = xr.open_dataset(bfile, mask_and_scale=False)
    ds = ds.sel(time=slice('2011-08-27', '2011-08-30'))
    buoy_lat, buoy_lon = ds.latitude.values[0], ds.longitude.values[0]

    buoy_data = np.squeeze(ds[vname])
    buoy_data[buoy_data == buoy_data._FillValue] = np.nan  # convert fill values to nans
    buoy_tm = buoy_data.time.values
    if 'wspd' in plot_var:  # estimate winds at 10m for buoy data only
        Zr = anemometer_height[buoy]  # height of anemometer on buoy
        Z = 10  # estimated height
        alpha = 0.143
        buoy_data = buoy_data * (Z / Zr) ** alpha
    buoy_values = buoy_data.values
    buoy_interp = buoy_data.interp(time=interp_time)
    buoy_interp['Time'] = buoy_interp['time']
    buoy_interp = buoy_interp.swap_dims({'time': 'Time'})  # make sure Time is the dimension for comparing to WRF
    buoy_interp_stdev = buoy_interp.std().values.item()

    append_obs(data_dict, buoy, plot_var, buoy_tm, buoy_values, buoy_interp.values, buoy_interp_stdev)

    return buoy_lat, buoy_lon, buoy_interp


def get_glider_data(plot_var, vname, data_dict):
    #ds = xr.open_dataset(glider_fname, mask_and_scale=False)
    # get the glider dataset
    ru_server = 'http://slocum-data.marine.rutgers.edu//erddap'
    id = 'ru16-20110810T1330-profile-sci-rt'
    glider_vars = ['latitude', 'longitude', 'depth', 'conductivity', 'density', 'salinity', 'sci_water_pressure',
                   'temperature', 'water_depth']
    ds = get_erddap_dataset(ru_server, 'tabledap', 'nc', id, glider_vars)
    ds = ds.swap_dims({'obs': 'time'})
    ds = ds.sortby(ds.time)
    ds = ds.sel(time=slice('2011-08-27T06', '2011-08-29T00'))
    glider_lat = 39.222263788799964
    glider_lon = -73.83885219215215
    gliderdata = ds[vname]
    #gliderdata[gliderdata == gliderdata._FillValue] = np.nan  # convert fill values to nans
    pressure = ds.sci_water_pressure
    #pressure[pressure == pressure._FillValue] = np.nan
    press_idx = pressure.values < .3  # select where pressure data are < 3 dbar
    gliderdata = gliderdata[press_idx]
    gliderdata = gliderdata[~np.isnan(gliderdata)]  # remove nans
    glider_tm = gliderdata.time.values
    glider_values = gliderdata.values
    glider_interp = gliderdata.resample(time='1H').mean()  # calculate hourly averages to match WRF data
    glider_interp['Time'] = glider_interp['time']
    glider_interp = glider_interp.swap_dims({'time': 'Time'})  # make sure Time is the dimension for comparing to WRF
    glider_interp_stdev = glider_interp.std().values.item()
    # ga = abs(ds.latitude - obs_lat) + abs(ds.longitude - obs_lon)
    # glider_idx = np.unravel_index(ga.argmin(), ga.shape)

    append_obs(data_dict, 'glider', plot_var, glider_tm, glider_values, glider_interp.values, glider_interp_stdev)

    return glider_lat, glider_lon, glider_interp


def get_met_data(met_fname, plot_var, vname, interp_time, data_dict):
    ds = pd.read_csv(met_fname)
    ds['tm'] = ds['time_stamp'].map(lambda t: t.split('.')[0])  # format time stamp
    met_lat = 39.52
    met_lon = -74.32

    met_values = np.array(ds[vname])
    if plot_var == 'airtemp':
        met_values = (met_values - 32) * 5 / 9  # convert F to C
    met_tm = np.array(pd.to_datetime(ds['tm']))
    metdata = xr.DataArray(met_values, coords=[met_tm], dims=['Time'])  # make sure Time is the dimension for comparing to WRF
    met_interp = metdata.interp(Time=interp_time)
    met_interp_stdev = met_interp.std().values.item()

    append_obs(data_dict, 'met_tower', plot_var, met_tm, metdata.values, met_interp.values, met_interp_stdev)

    return met_lat, met_lon, met_interp


def get_roms_time(roms_fname):
    ds = xr.open_dataset(roms_fname, mask_and_scale=False)
    wt = ds.ocean_time.values
    return wt


def get_wrf_time(wrf_fname):
    ds = xr.open_dataset(wrf_fname, mask_and_scale=False)
    wt = ds.Time.values
    return wt


def subset_roms(romsfiles, plot_var, vname, loci_idx, locj_idx, data_dict, compare_obs, interp_obs_values):
    for rf in romsfiles:
        roms_ver = 'roms' + rf.split('/')[-1].split('_')[-1][3]
        ds = xr.open_dataset(rf, mask_and_scale=False)

        # subset the data at the observation location
        model_var = ds[vname][:, :, loci_idx, locj_idx]
        h = ds.h[loci_idx, locj_idx]

        # grab SST - the max s_rho is the surface data
        model_var = model_var.sel(s_rho=np.nanmax(model_var.s_rho))

        # add ROMS data to dictionary
        try:
            data_dict[compare_obs][plot_var][roms_ver]
        except KeyError:
            data_dict[compare_obs][plot_var][roms_ver] = {}
        data_dict[compare_obs][plot_var][roms_ver]['roms_values'] = model_var.values

        # calculate stats and add values to dictionary
        data_dict[compare_obs][plot_var][roms_ver]['roms_std'] = model_var.std().values.item()
        data_dict[compare_obs][plot_var][roms_ver]['corr'] = cf.pearson_correlation(interp_obs_values, model_var)
        data_dict[compare_obs][plot_var][roms_ver]['crmse'] = cf.crmse(interp_obs_values, model_var)
        data_dict[compare_obs][plot_var][roms_ver]['rmse'] = cf.rmse(interp_obs_values, model_var)
        data_dict[compare_obs][plot_var][roms_ver]['bias'] = cf.model_bias(interp_obs_values, model_var)


def subset_wrf(wrffiles, plot_var, vname, loci_idx, locj_idx, data_dict, compare_obs, interp_obs_values):
    for wf in wrffiles:
        wrf_ver = 'wrf' + wf.split('/')[-1].split('_')[-2][-1]
        ds = xr.open_dataset(wf, mask_and_scale=False)

        # compare to WRF data
        if plot_var in ['wspd', 'wspd_adj']:  # calculate wind speed
            model_var = cf.wind_uv_to_spd(ds.U10, ds.V10)
        else:
            model_var = ds[vname]

        # subset the data at the observation location
        model_var = model_var[:, loci_idx, locj_idx]

        if plot_var in ['sst', 'airtemp']:
            model_var = model_var - 273.15  # convert to degrees C

        # add WRF data to dictionary
        try:
            data_dict[compare_obs][plot_var][wrf_ver]
        except KeyError:
            data_dict[compare_obs][plot_var][wrf_ver] = {}
        data_dict[compare_obs][plot_var][wrf_ver]['wrf_values'] = model_var.values

        # calculate stats and add values to dictionary
        data_dict[compare_obs][plot_var][wrf_ver]['wrf_std'] = model_var.std().values.item()
        data_dict[compare_obs][plot_var][wrf_ver]['corr'] = cf.pearson_correlation(interp_obs_values, model_var)
        data_dict[compare_obs][plot_var][wrf_ver]['crmse'] = cf.crmse(interp_obs_values, model_var)
        data_dict[compare_obs][plot_var][wrf_ver]['rmse'] = cf.rmse(interp_obs_values, model_var)
        data_dict[compare_obs][plot_var][wrf_ver]['bias'] = cf.model_bias(interp_obs_values, model_var)


def main(ddir):
    save_dir = os.path.join(os.path.dirname(ddir), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    #glider = '/Users/garzio/Documents/rucool/Miles/NOAA_COMT/data/glider/ru16-20110810T1330-profile-sci-rt_1da3_c3e1_51f4.nc'

    met_tower = '{}/met_tower.csv'.format(os.path.join(ddir, 'met_tower'))

    # define the plotting variable names: [buoy varname, WRF varname, mettower varname]
    plt_vars = {'sst': ['sea_surface_temperature', 'SST', 'temperature'], 'airtemp': ['air_temperature', 'T2', 'air_temp'],
                'slp': ['air_pressure', 'SLP', 'pressure'], 'wspd': ['wind_spd', 'U10', 'average']}

    # get the plotting colors and labels: [label, color]
    with open('plt_labels.pickle', 'rb') as handle:
        plt_labs = pickle.load(handle)

    # get observation data
    observations = ['glider', 'met_tower', '44009', '44065']
    #adjust_lon = {'44009': -75.0, '44065': -73.932889, 'met_tower': -74.32, 'glider': -73.83885219215215}
    ddict = {}

    wrf_files = sorted(glob.glob(os.path.join(ddir, '*subset.nc')))
    wrf_time = get_wrf_time(wrf_files[0])
    ddict['wrf_tm'] = wrf_time

    roms_files = sorted(glob.glob(os.path.join(ddir, 'roms*.nc')))
    roms_time = get_roms_time(roms_files[0])
    ddict['roms_tm'] = roms_time

    for obs in observations:
        ddict[obs] = {}
        for pv, varnames in plt_vars.items():
            if obs in ['44009', '44065']:
                buoy_fname = 'h'.join((obs, '2011.nc'))
                # buoydap = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/{}/{}{}'.format(buoy, buoy_fname, '.nc')
                buoy_file = os.path.join(ddir, 'buoys', buoy_fname)
                obs_lat, obs_lon, obs_interp = get_buoy_data(buoy_file, obs, pv, varnames[0], wrf_time, ddict)

                # for the first WRF file, find the lat/lon coordinate that corresponds with the observation lat/lon
                loci, locj = find_obs_loc(wrf_files[0], [obs_lat, obs_lon])

                # get WRF data corresponding to the observation lat/lon
                subset_wrf(wrf_files, pv, varnames[1], loci, locj, ddict, obs, obs_interp)

                if pv == 'sst':
                    # for the first ROMS file, find the lat/lon coordinate that corresponds with the observation lat/lon
                    loci, locj = find_obs_loc_roms(roms_files[0], [obs_lat, obs_lon])

                    # get ROMS data corresponding to the observation lat/lon
                    subset_roms(roms_files, pv, 'temp', loci, locj, ddict, obs, obs_interp)

            elif obs == 'met_tower':
                if 'sst' not in pv:
                    obs_lat, obs_lon, obs_interp = get_met_data(met_tower, pv, varnames[2], wrf_time, ddict)

                    # for the first WRF file, find the lat/lon coordinate that corresponds with the observation lat/lon
                    loci, locj = find_obs_loc(wrf_files[0], [obs_lat, obs_lon])

                    # get WRF data corresponding to the observation lat/lon
                    subset_wrf(wrf_files, pv, varnames[1], loci, locj, ddict, obs, obs_interp)

            else:  # glider
                if pv == 'sst':
                    obs_lat, obs_lon, obs_interp = get_glider_data(pv, varnames[2], ddict)

                    # for the first WRF file, find the lat/lon coordinate that corresponds with the observation lat/lon
                    loci, locj = find_obs_loc(wrf_files[0], [obs_lat, obs_lon])

                    # get WRF data corresponding to the observation lat/lon
                    subset_wrf(wrf_files, pv, varnames[1], loci, locj, ddict, obs, obs_interp)

                    # for the first ROMS file, find the lat/lon coordinate that corresponds with the observation lat/lon
                    loci, locj = find_obs_loc_roms(roms_files[0], [obs_lat, obs_lon])

                    # get ROMS data corresponding to the observation lat/lon
                    subset_roms(roms_files, pv, 'temp', loci, locj, ddict, obs, obs_interp)

    # plot timeseries panels of observations vs WRF
    fig, axs = plt.subplots(4, 3, figsize=(18, 12), sharex=True)
    ax1 = axs[0, 0]  # 44009 sst
    ax2 = axs[0, 1]  # 44065 sst
    ax3 = axs[0, 2]  # glider sst
    ax4 = axs[1, 0]  # 44009 airtemp
    ax5 = axs[1, 1]  # 44065 airtemp
    ax6 = axs[1, 2]  # met tower airtemp
    ax7 = axs[2, 0]  # 44009 slp
    ax8 = axs[2, 1]  # 44065 slp
    ax9 = axs[2, 2]  # met tower slp
    ax10 = axs[3, 0]  # 44009 wspd
    ax11 = axs[3, 1]  # 44065 wspd
    ax12 = axs[3, 2]  # met tower wspd
    ax1.set_title('Model vs Buoy 44009')
    ax2.set_title('Model vs Buoy 44065')
    ax3.set_title('Model vs RU-16 / Tuckerton Met Tower')

    axis_keys = {'44009': {'sst': ax1, 'airtemp': ax4, 'slp': ax7, 'wspd': ax10},
                 '44065': {'sst': ax2, 'airtemp': ax5, 'slp': ax8, 'wspd': ax11},
                 'met_tower': {'sst': ax3, 'airtemp': ax6, 'slp': ax9, 'wspd': ax12},
                 'glider': {'sst': ax3}}

    # define y-axis limits
    ylims = {'sst': [16, 26], 'airtemp': [20, 28], 'slp': [950, 1025], 'wspd': [0, 28], 'wspd_adj': [0, 28]}

    # define annotation text and location
    anno_xval = dt.datetime(2011, 8, 27, hour=8)
    anno_yval = {'sst': 17.5, 'airtemp': 21, 'slp': 960, 'wspd': 22, 'wspd_adj': 22}
    anno_lab = {'44009': {'sst': 'A)', 'airtemp': 'D)', 'slp': 'G)', 'wspd': 'J)'},
                '44065': {'sst': 'B)', 'airtemp': 'E)', 'slp': 'H)', 'wspd': 'K)'},
                'met_tower': {'sst': 'blank', 'airtemp': 'F)', 'slp': 'I)', 'wspd': 'L)'},
                'glider': {'sst': 'C)'}}

    wspd_rows = []
    wspd_adj_rows = []
    slp_rows = []
    df_headers = ['buoy', 'model', 'bias']
    #df_headers = ['buoy', 'model', 'rmse']

    for key, item in ddict.items():
        if key not in ['wrf_tm', 'roms_tm']:
            for key1, item1 in item.items():  # loop through each plotting variable
                if key1 not in ['lat', 'lon']:
                    axk = axis_keys[key][key1]  # get axis object
                    axk.plot(item1['obs_tm'], item1['obs_values'], color='k', lw=2, label='Obs')  # plot obs data
                    for key2, item2 in item1.items():
                        if 'wrf' in key2:
                            axk.plot(ddict['wrf_tm'], item2['wrf_values'], color=plt_labs[key2][1], lw=2,
                                     label=plt_labs[key2][0])
                            if key1 == 'wspd':
                                #wspd_rows.append([key, key2, item2['bias']])
                                wspd_rows.append([key, key2, item2['rmse']])
                            if key1 == 'wspd_adj':
                                wspd_adj_rows.append([key, key2, item2['bias']])
                            if key1 == 'slp':
                                #slp_rows.append([key, key2, item2['bias']])
                                slp_rows.append([key, key2, item2['rmse']])
                        elif 'roms' in key2:
                            axk.plot(ddict['roms_tm'], item2['roms_values'], color=plt_labs[key2][1], lw=2, ls='--',
                                     label=plt_labs[key2][0])
                        if key2 == 'wrf6':
                            axk.annotate(anno_lab[key][key1], xy=(anno_xval, anno_yval[key1]))  # add text to plot
                            axk.set_ylim(ylims[key1])
                            if key1 == 'sst':
                                axk.set_yticks(np.linspace(16, 26, 6))

    # print windspeed bias
    # df_wspd = pd.DataFrame(wspd_rows, columns=df_headers)
    # df_wspd.to_csv(os.path.join(save_dir, 'wspd_bias.csv'), index=False)
    #
    # df_wspd_adj = pd.DataFrame(wspd_adj_rows, columns=df_headers)
    # df_wspd_adj.to_csv(os.path.join(save_dir, 'wspd_adjusted_bias.csv'), index=False)

    # print SLP bias
    # df_slp = pd.DataFrame(slp_rows, columns=df_headers)
    # df_slp.to_csv(os.path.join(save_dir, 'slp_bias.csv'), index=False)

    # print rmse
    # df_rmse = pd.DataFrame(wspd_rows, columns=df_headers)
    # df_rmse.to_csv(os.path.join(save_dir, 'wspd_rmse.csv'), index=False)

    # df_rmse_slp = pd.DataFrame(slp_rows, columns=df_headers)
    # df_rmse_slp.to_csv(os.path.join(save_dir, 'slp_rmse-test.csv'), index=False)

    # format x-axes
    custom_xlim = ([dt.datetime(2011, 8, 27, hour=6), dt.datetime(2011, 8, 29, hour=0)])
    plt.setp(fig.get_axes(), xlim=custom_xlim)
    plt.setp(ax1.get_xticklabels(), fontsize=9)

    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(7))  # reduce number of x ticks

    # format y-axes
    yaxes = [ax2, ax3, ax5, ax6, ax8, ax9, ax11, ax12]
    for yax in yaxes:
        yax.set_yticklabels([])

    lpad = 16
    ax1.set_ylabel('SST ($^\circ$C)', labelpad=lpad)
    ax4.set_ylabel('Air Temperature ($^\circ$C)', labelpad=lpad)
    ax7.set_ylabel('Sea Level Pressure')
    ax10.set_ylabel('Wind Speed (m/s)', labelpad=lpad)

    ax6.legend(loc='upper right', framealpha=0.5, ncol=3, fontsize=10)

    # get legend handles for Taylor diagrams below
    handles, labels = plt.gca().get_legend_handles_labels()  # only show one set of legend labels
    by_label = dict(zip(labels, handles))
    # fig.legend(by_label.values(), by_label.keys(), framealpha=0.5, ncol=3, bbox_to_anchor=(0.89, 0.84), fontsize=12)

    plt.savefig(os.path.join(save_dir, 'wrf_obs_comp.png'), format='png', dpi=300)

    # plot Taylor diagrams (buoys and met tower)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    plt.text(0.08, 0.88, 'A)', fontsize=14, transform=plt.gcf().transFigure)  # sst
    plt.text(0.5, 0.88, 'B)', fontsize=14, transform=plt.gcf().transFigure)  # airtemp
    plt.text(0.08, 0.46, 'C)', fontsize=14, transform=plt.gcf().transFigure)  # slp
    plt.text(0.5, 0.46, 'D)', fontsize=14, transform=plt.gcf().transFigure)  # wspd and wspd

    plt.text(0.175, 0.91, 'Sea Surface Temperature', fontsize=14, transform=plt.gcf().transFigure)  # sst
    plt.text(0.64, 0.91, 'Air Temperature', fontsize=14, transform=plt.gcf().transFigure)  # airtemp
    plt.text(0.27, 0.475, 'MSLP', fontsize=14, transform=plt.gcf().transFigure)  # slp
    plt.text(0.66, 0.475, 'Wind Speed', fontsize=14, transform=plt.gcf().transFigure)  # wspd

    # turn off the subplot axis lines and labels
    for i, ax in enumerate(fig.axes):
        ax.axis('off')

    angle_lim = np.pi / 2
    std_lim = 1.75  # for normalized taylor diagram

    # for taylor contours
    rs, ts = np.meshgrid(np.linspace(0, std_lim), np.linspace(0, angle_lim))
    rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))

    # add taylor templates to figure
    fig, ax1 = cf.taylor_template(angle_lim, std_lim, 221, 'no', fig)  # sst
    fig, ax2 = cf.taylor_template(angle_lim, std_lim, 222, 'no', fig)  # airtemp
    fig, ax3 = cf.taylor_template(angle_lim, std_lim, 223, 'yes', fig)  # slp
    fig, ax4 = cf.taylor_template(angle_lim, std_lim, 224, 'yes', fig)  # wspd

    axis_keys = {'sst': ax1, 'airtemp': ax2, 'slp': ax3, 'wspd': ax4}

    # define marker shape for buoys
    marker_shp = {'44009': 's', '44065': 'o', 'met_tower': 'x', 'glider': '^'}

    for key, item in ddict.items():
        if 'wrf' not in key:
            for key1, item1 in item.items():  # loop through each plotting variable
                if key1 not in ['lat', 'lon']:
                    for key2, item2 in item1.items():
                        if 'wrf' in key2:
                            theta = np.arccos(item2['corr'])
                            rr = item2['wrf_std'] / item1['obs_interp_std']
                            # axis_keys[key1].plot(theta, rr, marker_shp[key], color=plt_labs[key2][1],
                            #                      markersize=8, mec='k')
                            if key1 == 'wspd_adj':
                                axis_keys[key1].plot(theta, rr, marker_shp[key], markersize=9,
                                                     color=plt_labs[key2][1])
                            else:
                                axis_keys[key1].plot(theta, rr, marker_shp[key], markersize=9, fillstyle='none',
                                                     mec=plt_labs[key2][1], markeredgewidth=2)

                    # add contours
                    axis_keys[key1].plot(0, 1, 'o', color='k', markersize=8, mec='k', alpha=1)
                    contours = axis_keys[key1].contour(ts, rs, rms, 3, colors='0.5')
                    plt.clabel(contours, inline=1, fontsize=10)

    # add data from IBTrACS comparison
    # minimum SLP
    ib_slp_stdev = 8.781468391418457
    ib_wind_stdev = 4.761341571807861
    with open('ibtracs_wrf_compare.pickle', 'rb') as handle:
        wrf_ibtracs = pickle.load(handle)
    for key, item in wrf_ibtracs.items():
        if 'wrf' in key:
            theta = np.arccos(item['minslp']['corr'])
            rr = item['minslp']['std'] / ib_slp_stdev
            ax3.plot(theta, rr, 'D', mec=plt_labs[key][1], markersize=9, fillstyle='none', markeredgewidth=2)

    # max windspeed
    for key, item in wrf_ibtracs.items():
        if 'wrf' in key:
            theta = np.arccos(item['maxws']['corr'])
            rr = item['maxws']['std'] / ib_wind_stdev
            ax4.plot(theta, rr, 'D', mec=plt_labs[key][1], markersize=9, fillstyle='none', markeredgewidth=2)

    # add legend from previous line plot to panel A
    fig.legend(by_label.values(), by_label.keys(), framealpha=0.5, ncol=2, bbox_to_anchor=(0.305, 0.829), fontsize=8)

    # add custom figure legend to panel C
    legend_elements = [Line2D([0], [0], marker='s', c='None', mec='dimgray', mew=2, ls='None', label='44009', ms=7),
                       Line2D([0], [0], marker='o', c='None', mec='dimgray', mew=2, ls='None', label='44065', ms=7),
                       Line2D([0], [0], marker='^', c='None', mec='dimgray', mew=2, ls='None', label='RU-16', ms=7),
                       Line2D([0], [0], marker='x', c='None', mec='dimgray', mew=2, ls='None', label='Tuckerton', ms=7),
                       Line2D([0], [0], marker='D', c='None', mec='dimgray', mew=2, ls='None', label='Best Track', ms=7)]

    #fig.legend(handles=legend_elements, framealpha=0.5, ncol=2, bbox_to_anchor=(0.321, 0.4), fontsize=8)
    fig.legend(handles=legend_elements, framealpha=0.5, ncol=2, bbox_to_anchor=(0.322, 0.405), fontsize=8)

    plt.savefig(os.path.join(save_dir, 'wrf_obs_taylor_normalized.png'), format='png', dpi=300)


if __name__ == '__main__':
    data_dir = '/Users/garzio/Documents/rucool/Miles/NOAA_COMT/data'  # location of data files
    main(data_dir)
