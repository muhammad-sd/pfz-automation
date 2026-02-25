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

# Define Area: Bay of Bengal (Example)
LAT_MIN, LAT_MAX = 17, 23
LON_MIN, LON_MAX = 88, 94

# Use 3-day buffer because NRT data is not instantaneous
target_dt = datetime.utcnow() - timedelta(days=3)
target_date_str = target_dt.strftime("%Y-%m-%d")

# JPL MUR SST (Standard, high-quality dataset)
SST_URL = "https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/MUR-JPL-L4-GLOB-v4.1.nc"

def generate_pfz():
    print(f"Starting pipeline for {target_date_str}...")
    
    try:
        # 2. DATA ACQUISITION
        ds = xr.open_dataset(SST_URL, chunks={}) 
        subset = ds.sel(
            time=target_date_str, 
            lat=slice(LAT_MIN, LAT_MAX), 
            lon=slice(LON_MIN, LON_MAX)
        ).load()

        # 3. CONVERT & SMOOTH
        sst_c = subset.analysed_sst.values - 273.15
        sst_smoothed = gaussian_filter(sst_c, sigma=1.5)
        
        # 4. GRADIENT CALCULATION (Thermal Fronts)
        dy, dx = np.gradient(sst_smoothed)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # PFZ Logic: Areas with highest 10% of thermal change
        threshold = np.nanpercentile(gradient_mag, 90)
        pfz_mask = np.where(gradient_mag > threshold, 1, np.nan)

        # 5. VISUALIZATION
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add Geography
        ax.add_feature(cfeature.LAND, facecolor='#dddddd')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        
        # Plot SST
        mesh = ax.pcolormesh(subset.lon, subset.lat, sst_c, 
                             cmap='RdYlBu_r', transform=ccrs.PlateCarree())
        plt.colorbar(mesh, label="Sea Surface Temperature (°C)", orientation='vertical', shrink=0.7)

        # Overlay PFZ
        y_idx, x_idx = np.where(pfz_mask == 1)
        ax.scatter(subset.lon[x_idx], subset.lat[y_idx], 
                   color='#00ff00', s=2, label='Potential Fishing Front', 
                   transform=ccrs.PlateCarree())

        plt.title(f"Potential Fishing Zones\nRegion: Bay of Bengal | Date: {target_date_str}")
        plt.legend(loc='lower right')

        # 6. SAVE
        # Always save as 'latest_pfz.png' for web hosting
        plt.savefig("outputs/latest_pfz.png", dpi=300, bbox_inches='tight')
        # Also save an archive version
        plt.savefig(f"outputs/pfz_{target_date_str}.png", dpi=150)
        
        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    generate_pfz()
