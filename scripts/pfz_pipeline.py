import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter

# -----------------------
# 1. SETTINGS & STABLE URLS
# -----------------------
# Using NOAA Multi-Scale Ultra-High Resolution (MUR) SST - Very stable
SST_URL = "https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/MUR-JPL-L4-GLOB-v4.1.nc"
# Using VIIRS Chlorophyll-a (7-day composite to avoid cloud gaps)
CHL_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla8day"

# Fetch data from 2 days ago to ensure "Near Real-Time" availability
target_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT12:00:00Z")

LAT_RANGE = slice(18, 24)
LON_RANGE = slice(88, 94)

try:
    print(f"Fetching data for {target_date}...")
    
    # 2. LOAD & SUBSET DATA
    ds_sst = xr.open_dataset(SST_URL).sel(time=target_date, lat=LAT_RANGE, lon=LON_RANGE)
    ds_chl = xr.open_dataset(CHL_URL).sel(time=target_date, latitude=LAT_RANGE, longitude=LON_RANGE)
    
    # Clean up Chlorophyll (rename and interpolate gaps)
    chl_data = ds_chl.chla.rename({"latitude": "lat", "longitude": "lon"}).interpolate_na(dim="lon")
    sst_data = ds_sst.analysed_sst - 273.15 # Convert Kelvin to Celsius

    # -----------------------
    # 3. GRADIENTS (Front Detection)
    # -----------------------
    # Smooth the data slightly to remove sensor noise
    sst_smoothed = gaussian_filter(sst_data.values, sigma=1)
    chl_smoothed = gaussian_filter(chl_data.values, sigma=1)

    # Calculate Gradients
    dy_s, dx_s = np.gradient(sst_smoothed)
    sst_grad = np.sqrt(dx_s**2 + dy_s**2)

    dy_c, dx_c = np.gradient(chl_smoothed)
    chl_grad = np.sqrt(dx_c**2 + dy_c**2)

    # -----------------------
    # 4. PFZ LOGIC (The Overlap)
    # -----------------------
    # Define thresholds: SST change > 0.3°C and High Chl-a gradient
    pfz = np.where((sst_grad > 0.3) & (chl_grad > np.nannanmedian(chl_grad)), 1, 0)

    # -----------------------
    # 5. VISUALIZATION
    # -----------------------
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black')
    
    # Plot SST as background
    im = ax.pcolormesh(ds_sst.lon, ds_sst.lat, sst_data, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(im, label="SST (°C)", orientation='horizontal', pad=0.05)

    # Overlay PFZ as bright points
    y_idx, x_idx = np.where(pfz == 1)
    ax.scatter(ds_sst.lon[x_idx], ds_sst.lat[y_idx], color='lime', s=2, label='PFZ (Potential Fishing Zone)')

    ax.set_title(f"PFZ Map: Bay of Bengal \n Date: {target_date[:10]}")
    plt.legend(loc='lower right')
    
    plt.show()

except Exception as e:
    print(f"Error: {e}. Try adjusting the 'target_date' or checking server status.")
