import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

# 1. DIRECTORY SETUP
os.makedirs("outputs", exist_ok=True)

# Define Area: Bay of Bengal
LAT_MIN, LAT_MAX = 17, 23
LON_MIN, LON_MAX = 88, 94

# 3-day buffer to ensure NRT data is finalized
target_dt = datetime.utcnow() - timedelta(days=3)
target_date_str = target_dt.strftime("%Y-%m-%d")

# CHANGED: Using a more stable NOAA CoastWatch ERDDAP link
# This dataset is "OISST v2.1" which is very reliable for automation
SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"

def generate_pfz():
    print(f"Starting pipeline for {target_date_str}...")
    
    try:
        # 2. DATA ACQUISITION with specific engine and timeout
        # Using decode_times=True to handle the netCDF time format automatically
        ds = xr.open_dataset(SST_URL, engine='netcdf4')
        
        print("Fetching SST slice...")
        subset = ds.sel(
            time=target_date_str, 
            latitude=slice(LAT_MIN, LAT_MAX), 
            longitude=slice(LON_MIN, LON_MAX),
            method='nearest'
        ).load()

        # 3. DATA PROCESSING
        # OISST is in Celsius already, but check variable names
        sst_vals = subset.sst.values[0] if len(subset.sst.shape) > 2 else subset.sst.values
        
        # Smooth to identify significant fronts
        sst_smoothed = gaussian_filter(sst_vals, sigma=1.2)
        
        # Gradient Magnitude
        dy, dx = np.gradient(sst_smoothed)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Identify PFZ (Top 10% gradients)
        threshold = np.nanpercentile(gradient_mag, 90)
        pfz_mask = np.where(gradient_mag > threshold, 1, np.nan)

        # 4. VISUALIZATION
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#cc9966')
        ax.add_feature(cfeature.COASTLINE)
        
        # Background SST
        mesh = ax.pcolormesh(subset.longitude, subset.latitude, sst_vals, 
                             cmap='RdYlBu_r', transform=ccrs.PlateCarree())
        plt.colorbar(mesh, label="SST (°C)", shrink=0.5)

        # Overlay PFZ
        y_idx, x_idx = np.where(pfz_mask == 1)
        ax.scatter(subset.longitude[x_idx], subset.latitude[y_idx], 
                   color='#00ff00', s=3, label='Potential Fishing Front', 
                   transform=ccrs.PlateCarree())

        plt.title(f"Potential Fishing Zones\n{target_date_str} | Bay of Bengal")
        plt.legend(loc='lower right')

        # 5. SAVE AND VERIFY
        out_file = "outputs/latest_pfz.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.savefig(f"outputs/pfz_{target_date_str}.png", dpi=150)
        
        if os.path.exists(out_file):
            print(f"SUCCESS: Map saved to {out_file}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    generate_pfz()
