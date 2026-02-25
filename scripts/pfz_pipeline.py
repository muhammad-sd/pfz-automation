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
LAT_MIN, LAT_MAX = 15, 24
LON_MIN, LON_MAX = 85, 94

# Use 4-day buffer to be safe (ERDDAP can be slow to update)
target_dt = datetime.utcnow() - timedelta(days=4)

# Stable NOAA OISST URL
SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"

def generate_pfz():
    print(f"Targeting approximate date: {target_dt.strftime('%Y-%m-%d')}")
    
    try:
        # 2. DATA ACQUISITION
        ds = xr.open_dataset(SST_URL, engine='netcdf4')
        
        print("Finding nearest time index...")
        # Step A: Select the nearest time first
        ds_time = ds.sel(time=target_dt, method='nearest')
        actual_date = str(ds_time.time.values)[:10]
        print(f"Actually using data from: {actual_date}")

        # Step B: Slice the spatial area
        print("Slicing spatial area...")
        subset = ds_time.sel(
            latitude=slice(LAT_MIN, LAT_MAX), 
            longitude=slice(LON_MIN, LON_MAX)
        ).load()

        # 3. DATA PROCESSING
        # Squeeze removes extra dimensions (like 'zlev')
        sst_vals = subset.sst.squeeze().values
        lons = subset.longitude.values
        lats = subset.latitude.values
        
        # Smooth the data
        sst_smoothed = gaussian_filter(sst_vals, sigma=1.2)
        
        # Calculate Gradient (Fronts)
        dy, dx = np.gradient(sst_smoothed)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # PFZ Logic (Top 10% of gradients)
        threshold = np.nanpercentile(gradient_mag, 90)
        pfz_mask = np.where(gradient_mag > threshold, 1, np.nan)

        # 4. VISUALIZATION
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='#cc9966', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)
        
        # Plot SST
        mesh = ax.pcolormesh(lons, lats, sst_vals, 
                             cmap='RdYlBu_r', transform=ccrs.PlateCarree())
        plt.colorbar(mesh, label="SST (°C)", shrink=0.5)

        # Overlay PFZ
        y_idx, x_idx = np.where(pfz_mask == 1)
        ax.scatter(lons[x_idx], lats[y_idx], 
                   color='#00ff00', s=100,alpha=0.6, label='Potential Fishing Area', 
                   transform=ccrs.PlateCarree(), zorder=4)

        plt.title(f"Potential Fishing Zones (PFZ)\nDate: {actual_date} | Bay of Bengal")
        plt.legend(loc='upper left')

        # 5. SAVE
        out_file = "outputs/latest_pfz.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.savefig(f"outputs/pfz_{actual_date}.png", dpi=150)
        
        if os.path.exists(out_file):
            print(f"SUCCESS: Map saved for {actual_date}")
            
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise e

if __name__ == "__main__":
    generate_pfz()
