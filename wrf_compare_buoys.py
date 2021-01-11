#!/usr/bin/env python

"""
Author: Lori Garzio on 1/11/2021
Last modified: 1/11/2021
"""

import numpy as np
import os
import glob
import datetime as dt
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import functions.common as cf


def main(ddir):
    save_dir = os.path.join(os.path.dirname(ddir), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # define the plotting variable names: [buoy varname, WRF varname]
    plt_vars = {'sst': ['sea_surface_temperature', 'SST'], 'airtemp': ['air_temperature', 'T2'],
                'wspd': ['wind_spd', 'U10'], 'slp': ['air_pressure', 'SLP']}

    # get the plotting colors and labels: [label, color]
    with open('plt_labels.pickle', 'rb') as handle:
        plt_labs = pickle.load(handle)

    # get buoy data
    buoys = ['44009', '44065']
    ddict = {}
    for buoy in buoys:
        buoy_fname = 'h'.join((buoy, '2011'))
        buoydap = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/{}/{}{}'.format(buoy, buoy_fname, '.nc')
        bds = xr.open_dataset(buoydap, mask_and_scale=False)
        bds = bds.sel(time=slice('2011-08-27', '2011-08-30'))

        ddict[buoy] = {}
        blat, blon = bds.latitude.values[0], bds.longitude.values[0]
        ddict[buoy]['lat'] = blat
        ddict[buoy]['lon'] = blon

        # get WRF data corresponding to the buoy lat/lon
        wrf_files = sorted(glob.glob(os.path.join(ddir, '*subset.nc')))
        for wf in wrf_files:
            fname = 'wrf' + wf.split('/')[-1].split('_')[-2][-1]
            ncfile = xr.open_dataset(wf, mask_and_scale=False)

            # for the first WRF file, find the lat/lon coordinate that corresponds with the buoy lat/lon
            if fname == 'wrf1':
                a = abs(ncfile.XLAT[:, :, 1] - blat) + abs(ncfile.XLONG[:, :, 1] - blon)
                i, j = np.unravel_index(a.argmin(), a.shape)
                wrf_time = ncfile.Time
                ddict['wrf_tm'] = wrf_time.values

            # interpolate the buoy data to WRF time and compare to the corresponding WRF data
            for pv, varnames in plt_vars.items():
                bv = np.squeeze(bds[varnames[0]])
                bv[bv == bv._FillValue] = np.nan  # convert fill values to nans
                v_interp = bv.interp(time=wrf_time)
                v_interp_stdev = v_interp.std().values.item()

                # add interpolated buoy data (raw and interpolated) to dictionary
                try:
                    ddict[buoy][pv]
                except KeyError:
                    ddict[buoy][pv] = {}
                    ddict[buoy][pv]['buoy_tm'] = bv.time.values
                    ddict[buoy][pv]['buoy_values'] = bv.values
                    ddict[buoy][pv]['buoy_values_interp'] = v_interp.values
                    ddict[buoy][pv]['buoy_interp_std'] = v_interp_stdev

                # compare to WRF data
                if varnames[1] == 'U10':  # calculate wind speed
                    wv = cf.wind_uv_to_spd(ncfile.U10, ncfile.V10)
                else:
                    wv = ncfile[varnames[1]]

                # subset the data at the buoy location
                wv = wv[:, i, j]

                if pv in ['sst', 'airtemp']:
                    wv = wv - 273.15  # convert to degrees C

                # add WRF data to dictionary
                try:
                    ddict[buoy][pv][fname]
                except KeyError:
                    ddict[buoy][pv][fname] = {}
                ddict[buoy][pv][fname]['wrf_values'] = wv.values

                # calculate stats and add values to dictionary
                ddict[buoy][pv][fname]['wrf_std'] = wv.std().values.item()
                ddict[buoy][pv][fname]['corr'] = cf.pearson_correlation(v_interp, wv)
                ddict[buoy][pv][fname]['crmse'] = cf.crmse(v_interp, wv)
                ddict[buoy][pv][fname]['bias'] = cf.model_bias(v_interp, wv)

    # plot timeseries of buoy vs model
    fig, axs = plt.subplots(4, 2, figsize=(16, 8), sharex=True)
    ax1 = axs[0, 0]  # 44009 sst
    ax2 = axs[0, 1]  # 44065 sst
    ax3 = axs[1, 0]  # 44009 airtemp
    ax4 = axs[1, 1]  # 44065 airtemp
    ax5 = axs[2, 0]  # 44009 wspd
    ax6 = axs[2, 1]  # 44065 wspd
    ax7 = axs[3, 0]  # 44009 slp
    ax8 = axs[3, 1]  # 44065 slp
    ax1.set_title('Model vs Buoy 44009')
    ax2.set_title('Model vs Buoy 44065')

    axis_keys = {'44009': {'sst': ax1, 'airtemp': ax3, 'wspd': ax5, 'slp': ax7},
                 '44065': {'sst': ax2, 'airtemp': ax4, 'wspd': ax6, 'slp': ax8}}

    # define y-axis limits
    ylims = {'sst': [16, 26], 'airtemp': [20, 28], 'wspd': [0, 28], 'slp': [950, 1025]}

    # define annotation text and location
    anno_xval = dt.datetime(2011, 8, 27, hour=8)
    anno_yval = {'sst': 18, 'airtemp': 21, 'wspd': 22, 'slp': 960}
    anno_lab = {'44009': {'sst': 'A) SST', 'airtemp': 'C) 2m Air Temperature', 'wspd': 'E) 10m Wind Speed',
                          'slp': 'G) Sea Level Pressure'},
                '44065': {'sst': 'B) SST', 'airtemp': 'D) 2m Air Temperature', 'wspd': 'F) 10m Wind Speed',
                          'slp': 'H) Sea Level Pressure'}}

    for key, item in ddict.items():
        if 'wrf' not in key:
            for key1, item1 in item.items():  # loop through each plotting variable
                if key1 not in ['lat', 'lon']:
                    axk = axis_keys[key][key1]  # get axis object
                    axk.plot(item1['buoy_tm'], item1['buoy_values'], color='k', lw=2, label='Buoy')  # plot buoy data
                    for key2, item2 in item1.items():
                        if 'wrf' in key2:
                            axk.plot(ddict['wrf_tm'], item2['wrf_values'], color=plt_labs[key2][1], lw=2,
                                     label=plt_labs[key2][0])
                        if key2 == 'wrf6':
                            axk.annotate(anno_lab[key][key1], xy=(anno_xval, anno_yval[key1]))  # add text to plot
                            axk.set_ylim(ylims[key1])
                            if key1 == 'sst':
                                axk.set_yticks(np.linspace(16, 26, 6))

    custom_xlim = ([dt.datetime(2011, 8, 27, hour=6), dt.datetime(2011, 8, 29, hour=0)])
    plt.setp(fig.get_axes(), xlim=custom_xlim)
    plt.setp(ax7.get_xticklabels(), fontsize=9)
    plt.setp(ax8.get_xticklabels(), fontsize=9)

    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax7.xaxis.set_major_formatter(xfmt)
    ax8.xaxis.set_major_formatter(xfmt)

    ax4.legend(loc='upper right', framealpha=0.5, ncol=3)

    plt.savefig(os.path.join(save_dir, 'wrf_buoy_comp.png'), format='png', dpi=300)

    # plot Taylor diagrams
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    titles = ['SST', '2m Air Temperature', '10m Wind Speed', 'Sea Level Pressure']

    # turn off the subplot axis lines and labels
    for i, ax in enumerate(fig.axes):
        ax.axis('off')
        ax.set_title(titles[i])

    angle_lim = np.pi / 2
    std_lim = 1.75

    # for taylor contours
    rs, ts = np.meshgrid(np.linspace(0, std_lim), np.linspace(0, angle_lim))
    rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))

    # add taylor templates to figure
    fig, ax1 = cf.taylor_template(angle_lim, std_lim, 221, fig)  # sst
    fig, ax2 = cf.taylor_template(angle_lim, std_lim, 222, fig)  # airtemp
    fig, ax3 = cf.taylor_template(angle_lim, std_lim, 223, fig)  # wspd
    fig, ax4 = cf.taylor_template(angle_lim, std_lim, 224, fig)  # slp

    axis_keys = {'sst': ax1, 'airtemp': ax2, 'wspd': ax3, 'slp': ax4}

    # define marker shape for buoys
    marker_shp = {'44009': 's', '44065': 'o'}

    for key, item in ddict.items():
        if 'wrf' not in key:
            for key1, item1 in item.items():  # loop through each plotting variable
                if key1 not in ['lat', 'lon']:
                    for key2, item2 in item1.items():
                        if 'wrf' in key2:
                            theta = np.arccos(item2['corr'])
                            rr = item2['wrf_std'] / item1['buoy_interp_std']
                            if key1 == 'slp':  # for the last plot, extract labels for legend
                                axis_keys[key1].plot(theta, rr, marker_shp[key], color=plt_labs[key2][1],
                                                     label=plt_labs[key2][0], markersize=8, mec='k')
                            else:
                                axis_keys[key1].plot(theta, rr, marker_shp[key], color=plt_labs[key2][1],
                                                     markersize=8, mec='k')
                    # add contours
                    axis_keys[key1].plot(0, 1, 'D', color='y', markersize=8, mec='k', alpha=1)
                    contours = axis_keys[key1].contour(ts, rs, rms, 3, colors='0.5')
                    plt.clabel(contours, inline=1, fontsize=10)
                    axis_keys[key1].set_title('SST')

    plt.legend(loc='upper right', ncol=3, fontsize=8)

    plt.savefig(os.path.join(save_dir, 'buoy_wrf_taylor_normalized.png'), format='png', dpi=300)


if __name__ == '__main__':
    data_dir = '/Users/lgarzio/Documents/rucool/Miles/NOAA_COMT/data'  # location of wrf model data files
    main(data_dir)
