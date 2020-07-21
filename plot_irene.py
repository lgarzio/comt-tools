#!/usr/bin/env python

"""
Author: Lori Garzio on 7/9/2020
Last modified: 7/14/2020
Creates hourly images of wind speeds, SST, SLP, Latent and Sensible Heat Flux
"""

import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 13})


def add_map_features(ax, axes_limits, land_color=None):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    :param land_color: optional color for land
    """
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='dotted', x_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    #gl.rotate_labels = False
    gl.xpadding = 12
    gl.ypadding = 12
    ax.set_extent(axes_limits)

    if land_color:
        lc = land_color
    else:
        lc = 'none'

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor=lc)
    ax.add_feature(land, zorder=1)

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

    ax.add_feature(cfeature.BORDERS, zorder=6)
    ax.add_feature(state_lines, zorder=7, edgecolor='black')

    feature = cfeature.NaturalEarthFeature(
        name='coastline', category='physical',
        scale='10m',
        edgecolor='black', facecolor='none')
    ax.add_feature(feature, zorder=8)


def add_slp_contours(ax, londata, latdata, slp_data):
    contour_list = np.round(np.linspace(954, 1022, 9))
    CS = ax.contour(londata, latdata, slp_data, contour_list, colors='black', linewidths=.5, transform=ccrs.PlateCarree())
    ax.clabel(CS, inline=True, fontsize=10.5, fmt='%d')


def plot_contourf(fig, ax, ttl, lon_data, lat_data, var_data, clevs, cmap, clab):
    """
    Create a filled contour plot with user-defined levels and colors
    :param fig: figure object
    :param ax: plotting axis object
    :param ttl: plot title
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param clevs: list of colorbar level demarcations
    :param cmap: colormap
    :param norm: object that normalizes the colorbar level demarcations
    :param clab: colorbar label
    :returns fig, ax objects
    """
    plt.subplots_adjust(right=0.88)
    plt.title(ttl)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    # cs = ax.contourf(lon_data, lat_data, var_data, clevs, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), alpha=.9)
    cs = ax.contourf(lon_data, lat_data, var_data, clevs, cmap=cmap, transform=ccrs.PlateCarree(), alpha=.9)
    cb = plt.colorbar(cs, cax=cax)
    cb.set_label(label=clab)

    return fig, ax


def plot_pcolor(fig, ax, ttl, lon_data, lat_data, var_data, var_min, var_max, cmap, clab):
    """
    Create a pseudocolor plot
    :param fig: figure object
    :param ax: plotting axis object
    :param ttl: plot title
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param var_min: minimum value for plotting (for fixed colorbar)
    :param var_max: maximum value for plotting (for fixed colorbar)
    :param cmap: color map
    :param clab: colorbar label
    :returns fig, ax objects
    """
    plt.subplots_adjust(right=0.88)
    plt.title(ttl)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    h = ax.pcolor(lon_data, lat_data, var_data, vmin=var_min, vmax=var_max, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
    # h = ax.pcolor(lon_data, lat_data, var_data, cmap=cmap, transform=ccrs.PlateCarree())

    plt.colorbar(h, cax=cax, label=clab)

    return fig, ax


def plt_heatflux(nc, varname, model, ax_lims, save_dir):
    """
    Create a pcolor surface map of heat flux
    :param nc: netcdf file
    :paran varname: variable string name to plot
    :param model: the model version that is being plotted
    :param ax_lims: axis limits for plotting
    :param figname: full file path to save directory and save filename
    """
    flx = nc[varname]
    if varname == 'LH':
        var = 'Latent Heat Flux'
        datarange = [-700, 700]
    else:
        var = 'Upward Heat Flux'
        datarange = [-400, 400]
    color_label = '{} (W m-2)'.format(var)

    slp = nc['SLP']

    for i, t in enumerate(nc['Time'].values):
        print('Plotting {}'.format(pd.to_datetime(t).strftime('%Y%m%dT%H%M')))
        flx_sub = flx.sel(Time=t)

        fig, ax, lat, lon = set_map(flx_sub)

        figttl = '{}, Sea Level Pressure (mb)\n{} UTC'.format(var, pd.to_datetime(t).strftime('%b %d %Y %H:%M'))

        fig, ax = plot_pcolor(fig, ax, figttl, lon, lat, flx_sub, datarange[0], datarange[1], 'coolwarm', color_label)

        add_map_features(ax, ax_lims)

        # add contour lines for SLP
        slp_sub = slp.sel(Time=t)
        add_slp_contours(ax, lon, lat, slp_sub)

        sfile = '{}_{}_{}.png'.format(model, varname, pd.to_datetime(t).strftime('%Y%m%dT%H%M'))
        plt.savefig(os.path.join(save_dir, sfile), dpi=200)
        plt.close()


def plt_slp(nc, model, ax_lims, save_dir):
    """
    Create a contour surface map of sea level pressure
    :param nc: netcdf file
    :param model: the model version that is being plotted
    :param ax_lims: axis limits for plotting
    :param figname: full file path to save directory and save filename
    """
    slp = nc['SLP']
    color_label = 'Sea Level Pressure (mb)'
    levels = np.round(np.linspace(954, 1022, 9))
    #levels = np.round(np.linspace(954, 1022, 12))

    for i, t in enumerate(nc['Time'].values):
        print('Plotting {}'.format(pd.to_datetime(t).strftime('%Y%m%dT%H%M')))
        slp_sub = slp.sel(Time=t)

        fig, ax, lat, lon = set_map(slp_sub)
        figttl = 'Sea Level Pressure: {} UTC'.format(pd.to_datetime(t).strftime('%b %d %Y %H:%M'))

        plot_contourf(fig, ax, figttl, lon, lat, slp_sub, levels, 'jet', color_label)

        # contour_list = [940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008,
        #                 1012, 1016, 1020, 1024, 1028, 1032, 1036, 1040]
        # CS = ax.contour(lon, lat, slp_sub, levels, colors='black', linewidths=.5, transform=ccrs.PlateCarree())
        # ax.clabel(CS, inline=True, fontsize=10.5, fmt='%d')
        # plt.title(figttl)

        add_map_features(ax, ax_lims)

        sfile = '{}_slp_{}.png'.format(model, pd.to_datetime(t).strftime('%Y%m%dT%H%M'))
        plt.savefig(os.path.join(save_dir, sfile), dpi=200)
        plt.close()


def plt_sst(nc, model, ax_lims, save_dir):
    """
    Create a pcolor surface map of sea surface temperature
    :param nc: netcdf file
    :param model: the model version that is being plotted
    :param ax_lims: axis limits for plotting
    :param figname: full file path to save directory and save filename
    """
    sst = nc['SST']
    landmask = nc['LANDMASK']  # 1=land, 0=water
    lakemask = nc['LAKEMASK']  # 1=lake, 0=non-lake
    color_label = 'SST ($^\circ$C)'

    slp = nc['SLP']

    for i, t in enumerate(nc['Time'].values):
        print('Plotting {}'.format(pd.to_datetime(t).strftime('%Y%m%dT%H%M')))
        sst_sub = sst.sel(Time=t)
        landmask_sub = landmask.sel(Time=t).values
        lakemask_sub = lakemask.sel(Time=t).values

        ldmask = np.logical_and(landmask_sub == 1, landmask_sub == 1)
        sst_sub.values[ldmask] = np.nan  # turn values over land to nans

        lkmask = np.logical_and(lakemask_sub == 1, lakemask_sub == 1)
        sst_sub.values[lkmask] = np.nan  # turn values over lakes to nans

        fig, ax, lat, lon = set_map(sst_sub)

        # convert degrees K to degrees C
        sst_sub_c = sst_sub.values - 273.15

        figttl = 'Sea Surface Temperature, Sea Level Pressure (mb)\n{} UTC'.format(pd.to_datetime(t).strftime('%b %d %Y %H:%M'))

        fig, ax = plot_pcolor(fig, ax, figttl, lon, lat, sst_sub_c, 0, 40, 'jet', color_label)

        add_map_features(ax, ax_lims, 'gray')
        #add_map_features(ax, ax_lims)

        # add contour lines for SLP
        slp_sub = slp.sel(Time=t)
        add_slp_contours(ax, lon, lat, slp_sub)

        sfile = '{}_sst_{}.png'.format(model, pd.to_datetime(t).strftime('%Y%m%dT%H%M'))
        plt.savefig(os.path.join(save_dir, sfile), dpi=200)
        plt.close()


def plt_windsp(nc, model, ht, ax_lims, save_dir):
    """
    Create pseudocolor surface maps of wind speed with quivers indicating wind direction.
    :param nc: netcdf file
    :param model: the model version that is being plotted
    :param ht: wind speed height to plot, e.g. 10, 150
    :param ax_lims: axis limits for plotting
    :param save_dir: path to save file directory
    """
    if ht == 10:
        u = nc['U10']
        v = nc['V10']
    else:
        u = nc.sel(height=ht)['U']
        v = nc.sel(height=ht)['V']

    color_label = 'Wind Speed (m/s)'

    slp = nc['SLP']

    for i, t in enumerate(nc['Time'].values):
        print('Plotting {}'.format(pd.to_datetime(t).strftime('%Y%m%dT%H%M')))
        u_sub = u.sel(Time=t)
        v_sub = v.sel(Time=t)

        fig, ax, lat, lon = set_map(u_sub)

        # calculate wind speed from u and v
        speed = wind_uv_to_spd(u_sub, v_sub)

        figttl = '{}m Wind Speeds, Sea Level Pressure (mb)\n{} UTC'.format(ht, pd.to_datetime(t).strftime('%b %d %Y %H:%M'))

        fig, ax = plot_pcolor(fig, ax, figttl, lon, lat, speed.values, 0, 40, 'BuPu', color_label)

        # subset the quivers and add as a layer
        qs = 9
        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_sub.values[::qs, ::qs], v_sub.values[::qs, ::qs], scale=1000,
                  width=.002, headlength=4, transform=ccrs.PlateCarree(), zorder=3)

        add_map_features(ax, ax_lims)

        # add contour lines for SLP
        slp_sub = slp.sel(Time=t)
        add_slp_contours(ax, lon, lat, slp_sub)

        sfile = '{}_wndsp10_{}.png'.format(model, pd.to_datetime(t).strftime('%Y%m%dT%H%M'))
        plt.savefig(os.path.join(save_dir, sfile), dpi=200)
        plt.close()


def set_map(data):
    """
    Set up the map and projection
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :returns fig, ax objects
    :returns dlat: latitude data
    returns dlon: longitude data
    """
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 9), subplot_kw=dict(projection=lccproj))
    #fig, ax = plt.subplots(figsize=(8, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))

    dlat = data['XLAT'].values
    dlon = data['XLONG'].values

    return fig, ax, dlat, dlon


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    :param u: west/east direction (wind from the west is positive, from the east is negative)
    :param v: south/noth direction (wind from the south is positive, from the north is negative)
    :returns WSPD: wind speed calculated from the u and v wind components
    """
    WSPD = np.sqrt(np.square(u)+np.square(v))

    return WSPD


def main(f):
    sDir = os.path.dirname(os.path.dirname(f))
    ncfile = xr.open_dataset(f, mask_and_scale=False)

    # plt_vars = ['U10', 'SST', 'SLP', 'LH', 'HFX']
    plt_vars = ['HFX']

    axlims = [-80, -61, 32.3, 46]

    for pv in plt_vars:
        if pv == 'U10':
            plt_windsp(ncfile, 'irene_wrf_his', 10, axlims, sDir)
        elif pv == 'SST':
            plt_sst(ncfile, 'irene_wrf_his', axlims, sDir)
        elif pv == 'SLP':
            plt_slp(ncfile, 'irene_wrf_his', axlims, sDir)
        elif pv in ['LH', 'HFX']:
            plt_heatflux(ncfile, pv, 'irene_wrf_his', axlims, sDir)


if __name__ == '__main__':
    fpath = '/Users/lgarzio/Documents/rucool/Miles/NOAA_COMT/irene/data/irene_wrf_his_d01_2011-08-27_06_00_00_subset.nc'  # on local machine
    # fpath = '/home/lgarzio/rucool/NOAA_COMT/data/july10/irene_wrf_his_d01_2011-08-27_06_00_00.nc'  # on server
    main(fpath)
