#!/usr/bin/env python

"""
Author: Mike Smith
Modified on July 14, 2020 by Lori Garzio
Last modified July 21, 2020
"""
import numpy as np
import os
import pandas as pd
import xarray as xr
from collections import OrderedDict
from wrf import getvar, ALL_TIMES


def delete_attr(da):
    """
    Delete these local attributes because they are not necessary
    :param da: DataArray of variable
    :return: DataArray of variable with local attributes removed
    """

    for item in ['projection', 'coordinates', 'MemoryOrder', 'FieldType', 'stagger', 'missing_value']:
        try:
            del da.attrs[item]
        except KeyError:
            continue
    return da


def main(fpath):
    fname = fpath.split('/')[-1]
    save_file = os.path.join(os.path.dirname(fpath), '{}_subset.nc'.format(fname.split('.')[0]))

    # List of variables that are already included in the WRF output and that we want to compute using the wrf-python
    variables = dict(
        primary=['XLAT', 'XLONG', 'LANDMASK', 'LAKEMASK', 'SST', 'LH', 'HFX', 'U10', 'V10', 'T2'],
        computed=['slp', 'rh2']
    )

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Open using netCDF toolbox
    ncfile = xr.open_dataset(fpath)
    original_global_attributes = ncfile.attrs
    ncobj = ncfile._file_obj.ds

    # Load primary variables and append to list
    primary_vars = {}

    for var in variables['primary']:
        primary_vars[var] = delete_attr(getvar(ncobj, var, timeidx=ALL_TIMES))

    # Calculate diagnostic variables defined above and add to dictionary
    diagnostic_vars = {}

    for dvar in variables['computed']:
        diagnostic_vars[dvar.upper()] = delete_attr(getvar(ncobj, dvar, timeidx=ALL_TIMES))

    # Create xarray dataset of primary and diagnostic variables
    ds = xr.Dataset({**primary_vars, **diagnostic_vars})

    # Add description and units for lon, lat dimensions for georeferencing
    ds['XLAT'].attrs['description'] = 'latitude'
    ds['XLAT'].attrs['units'] = 'degree_north'
    ds['XLONG'].attrs['description'] = 'longitude'
    ds['XLONG'].attrs['units'] = 'degree_east'

    # Set time attribute
    ds['Time'].attrs['standard_name'] = 'time'

    # Set XTIME attribute
    ds['XTIME'].attrs['units'] = 'minutes'

    # Set lon attributes
    ds['XLONG'].attrs['long_name'] = 'Longitude'
    ds['XLONG'].attrs['standard_name'] = 'longitude'
    ds['XLONG'].attrs['short_name'] = 'lon'
    ds['XLONG'].attrs['units'] = 'degrees_east'
    ds['XLONG'].attrs['axis'] = 'X'
    ds['XLONG'].attrs['valid_min'] = np.float32(-180.0)
    ds['XLONG'].attrs['valid_max'] = np.float32(180.0)

    # Set lat attributes
    ds['XLAT'].attrs['long_name'] = 'Latitude'
    ds['XLAT'].attrs['standard_name'] = 'latitude'
    ds['XLAT'].attrs['short_name'] = 'lat'
    ds['XLAT'].attrs['units'] = 'degrees_north'
    ds['XLAT'].attrs['axis'] = 'Y'
    ds['XLAT'].attrs['valid_min'] = np.float32(-90.0)
    ds['XLAT'].attrs['valid_max'] = np.float32(90.0)

    ds['T2'].attrs['long_name'] = 'Air Temperature at 2m'

    datetime_format = '%Y%m%dT%H%M%SZ'
    created = pd.Timestamp(pd.datetime.utcnow()).strftime(datetime_format)  # creation time Timestamp
    time_start = pd.Timestamp(pd.Timestamp(np.min(ds.Time.data))).strftime(datetime_format)
    time_end = pd.Timestamp(pd.Timestamp(np.max(ds.Time.data))).strftime(datetime_format)
    global_attributes = OrderedDict([
        ('date_created', created),
        ('time_coverage_start', time_start),
        ('time_coverage_end', time_end)])

    global_attributes.update(original_global_attributes)
    ds = ds.assign_attrs(global_attributes)

    # Add compression to all variables
    encoding = {}
    for k in ds.data_vars:
        encoding[k] = {'zlib': True, 'complevel': 1}

    # add the encoding for time so xarray exports the proper time.
    # Also remove compression from dimensions. They should never have fill values
    encoding['Time'] = dict(calendar='gregorian', zlib=False, _FillValue=False, dtype=np.double)
    encoding['XLONG'] = dict(zlib=False, _FillValue=False)
    encoding['XLAT'] = dict(zlib=False, _FillValue=False)

    ds.to_netcdf(save_file, encoding=encoding, format='netCDF4', engine='netcdf4', unlimited_dims='Time')


if __name__ == '__main__':
    filepath = '/Users/lgarzio/Documents/rucool/Miles/NOAA_COMT/tsfay/20200715/fay_wrf_his_d01_2020-07-09_12_00_00.nc'
    main(filepath)
