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

# Use 3-day buffer
target_dt = datetime.utcnow() - timedelta(days=3)
target_date_str = target_dt.strftime("%Y-%m-%d")

# Stable NOAA OISST URL
SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"

def generate_pfz():
    print(f"Starting pipeline for {target_date_str}...")
    
    try:
        # 2. DATA ACQUISITION
        # Open dataset with netcdf4 engine
        ds = xr.open_dataset(SST_URL, engine='netcdf4')
        
        print("Fetching SST slice...")
        # Select the time first, then slice coordinates
        # We use slice() without method='nearest' for range selection
        subset = ds.sel(
            time=target_date_str, 
            latitude=slice(LAT_MIN, LAT_MAX), 
            longitude=slice(LON_MIN, LON_MAX)
        ).load()

        # 3. DATA PROCESSING
        # OISST sst variable usually has a 'zlev' (depth) dimension of size 1
        # We squeeze() to remove any single-dimensional entries (time, zlev)
        sst_vals = subset.sst.squeeze().values
        lons = subset.longitude.values
        lats = subset.latitude.values
        
        # Smooth the data to find real oceanographic fronts
        sst_smoothed = gaussian_filter(sst_vals, sigma=1.2)
        
        # Calculate Gradient (Fronts)
        dy, dx = np.gradient(sst_smoothed)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Identify PFZ (Top 10% gradients)
        threshold = np.nanpercentile(gradient_mag, 90)
        pfz_mask = np.where(gradient_mag > threshold, 1, np.nan)

        # 4. VISUALIZATION
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add high-resolution features
        ax.add_feature(cfeature.LAND, facecolor='#cc9966', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
        
        # Plot SST
        # Use pcolormesh for proper coordinate alignment
        mesh = ax.pcolormesh(lons, lats, sst_vals, 
                             cmap='RdYlBu_r', transform=ccrs.PlateCarree())
        plt.colorbar(mesh, label="SST (°C)", shrink=0.5, pad=0.02)

        # Overlay PFZ (Green points)
        y_idx, x_idx = np.where(pfz_mask == 1)
        ax.scatter(lons[x_idx], lats[y_idx], 
                   color='#00ff00', s=5, label='Potential Fishing Front', 
                   transform=ccrs.PlateCarree(), zorder=4)

        plt.title(f"Potential Fishing Zones (PFZ)\nDate: {target_date_str} | Region: Bay of Bengal", fontsize=12)
        plt.legend(loc='lower right')

        # 5. SAVE AND VERIFY
        out_file = "outputs/latest_pfz.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.savefig(f"outputs/pfz_{target_date_str}.png", dpi=150)
        
        if os.path.exists(out_file):
            print(f"SUCCESS: Map saved to {out_file}")
            
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        # Re-raise error to make GitHub Action fail (RED)
        raise e

if __name__ == "__main__":
    generate_pfz()
