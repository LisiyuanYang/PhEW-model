import numpy as np
from scipy.interpolate import RegularGridInterpolator
import glob
from scanf import scanf
import h5py
from phew_util import XH, XHE, MHYDR

cooling_table_path = '~/Astronomy/CoolingTables/'
collisional_cooling_table_file = cooling_table_path + 'z_collis.hdf5'
_collisional_cooling_table_nometal = None
_collisional_cooling_table_metal = None
_temp_bins = None
_photo_nometal_interpolator = None
_photo_metal_interpolator = None

photo_cooling_files = glob.glob('z_*.[0-9][0-9][0-9].hdf5', root_dir=cooling_table_path)
zlist = np.array([scanf('z_%f.hdf5', f)[0] for f in photo_cooling_files])

def collisional_cooling_rate(temperature, metallicity=1.):#Metallicity should be in solar metallicities
    global _collisional_cooling_table_nometal, _collisional_cooling_table_metal, _temp_bins
    if _collisional_cooling_table_metal is None:
        with h5py.File(collisional_cooling_table_file) as f:
            _collisional_cooling_table_nometal = f['Metal_free/Net_Cooling'][0, :]#The 0-th row corresponds to a He mass fraction of 23.8%
            _collisional_cooling_table_metal = f['Total_Metals/Net_cooling'][:]#Assumes solar abundance ratios
            _temp_bins = f['Metal_free/Temperature_bins'][:]
    metal_free_cooling = np.interp(temperature, _temp_bins, _collisional_cooling_table_nometal)
    metal_cooling = np.interp(temperature, _temp_bins, _collisional_cooling_table_metal) * metallicity
    return metal_free_cooling + metal_cooling

def collisional_cooling_time(temperature, density, metallicity=1.):#in Myr
    cooling_rate = collisional_cooling_rate(temperature, metallicity)
    return 1.098e-53 * (2 + 3 * XHE) / XH * temperature / (cooling_rate * density)

def _initialize_photo_interpolator():
    global _photo_nometal_interpolator, _photo_metal_interpolator
    nometal_cooling_table_list = []
    metal_cooling_table_list = []
    for i in range(len(zlist)):
        with h5py.File(cooling_table_path + photo_cooling_files[i]) as f:
            if i == 0:
                temp_bins = f['Metal_free/Temperature_bins'][:]
                nh_bins = f['Metal_free/Hydrogen_density_bins'][:]
            nometal_cooling_table_list.append(f['Metal_free/Net_Cooling'][0, :, :])
            metal_cooling_table_list.append(f['Total_Metals/Net_cooling'][:])
    nometal_cooling_table_list = np.array(nometal_cooling_table_list)
    metal_cooling_table_list = np.array(metal_cooling_table_list)
    _photo_nometal_interpolator = RegularGridInterpolator((zlist, temp_bins, nh_bins), nometal_cooling_table_list, bounds_error=False, fill_value=None)
    _photo_metal_interpolator = RegularGridInterpolator((zlist, temp_bins, nh_bins), metal_cooling_table_list, bounds_error=False, fill_value=None)

def photo_cooling_rate(temperature, density, redshift, metallicity=1.):
    global _photo_nometal_interpolator, _photo_metal_interpolator
    if _photo_metal_interpolator is None:
        _initialize_photo_interpolator()
    nh = density * XH / MHYDR
    redshift_list = np.broadcast_to(redshift, np.shape(temperature))#If redshift is an array, keep it unchanged
    points = np.array([redshift_list, temperature, nh], dtype=float).T
    metal_free_cooling = _photo_nometal_interpolator(points)
    metal_cooling = _photo_metal_interpolator(points) * metallicity
    return metal_free_cooling + metal_cooling

def photo_cooling_time(temperature, density, redshift, metallicity=1.):
    cooling_rate = photo_cooling_rate(temperature, density, redshift, metallicity)
    return 1.098e-53 * (2 + 3 * XHE) / XH * temperature / (cooling_rate * density)
