import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

# 1. SETUP
os.makedirs("outputs", exist_ok=True)

# Define Area: Bay of Bengal (Example)
LAT_MIN, LAT_MAX = 17, 23
LON_MIN, LON_MAX = 88, 94

# Use a 3-day delay to ensure the satellite data is available on the server
target_dt = datetime.utcnow() - timedelta(days=3)
target_date_str = target_dt.strftime("%Y-%m-%d")

# High-resolution SST (JPL MUR) - 0.01 degree resolution
SST_URL = "https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/MUR-JPL-L4-GLOB-v4.1.nc"

def generate_pfz():
    print(f"Attempting to fetch data for: {target_date_str}")
    
    try:
        # 2. DATA ACQUISITION
        # We only download the small slice we need to save memory
        ds = xr.open_dataset(SST_URL, chunks={}) 
        subset = ds.sel(
            time=target_date_str, 
            lat=slice(LAT_MIN, LAT_MAX), 
            lon=slice(LON_MIN, LON_MAX)
        ).load()

        # 3. CALCULATE THERMAL FRONTS
        # Convert Kelvin to Celsius
        sst_c = subset.analysed_sst.values - 273.15
        
        # Smooth to remove sensor noise
        sst_smoothed = gaussian_filter(sst_c, sigma=1.5)
        
        # Calculate Gradient Magnitude (Front Intensity)
        dy, dx = np.gradient(sst_smoothed)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Define PFZ: Where gradient is in the top 10% of the area
        threshold = np.nanpercentile(gradient_mag, 90)
        pfz_points = np.where(gradient_mag > threshold, 1, np.nan)

        # 4. MAPPING
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add high-res geographical features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot SST Background
        mesh = ax.pcolormesh(subset.lon, subset.lat, sst_c, 
                             cmap='Spectral_r', transform=ccrs.PlateCarree())
        plt.colorbar(mesh, label="Sea Surface Temperature (°C)", pad=0.02)

        # Overlay PFZ as dots
        # Converting indices back to lat/lon for scatter plot
        y_idx, x_idx = np.where(pfz_points == 1)
        ax.scatter(subset.lon[x_idx], subset.lat[y_idx], 
                   color='magenta', s=5, label='Potential Fishing Front', 
                   transform=ccrs.PlateCarree())

        plt.title(f"Potential Fishing Zones (PFZ)\nDate: {target_date_str}", fontsize=15)
        plt.legend(loc='lower right')

        # Save output
        output_path = "outputs/latest_pfz.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(f"outputs/pfz_{target_date_str}.png", dpi=150)
        
        print(f"Successfully generated: {output_path}")

    except Exception as e:
        print(f"Error encountered: {e}")

if __name__ == "__main__":
    generate_pfz()
