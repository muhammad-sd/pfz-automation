import os
import xarray as xr
import numpy as np
import folium
from folium import plugins, raster_layers
import branca.colormap as cm
from scipy.ndimage import gaussian_filter
from datetime import datetime
import requests
import time

# 1. SETUP
os.makedirs("outputs", exist_ok=True)
LAT_MIN, LAT_MAX = 14, 25
LON_MIN, LON_MAX = 85.5, 94

DATASET_ID = "ncdcOisst21Agg_LonPM180"
BASE_URL = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{DATASET_ID}"

def hex_to_rgba(hex_str, alpha=1.0):
    """Safely converts hex strings (including 8-char RGBA hex) to float tuples"""
    hex_str = hex_str.lstrip('#')
    # If branca returns RGBA hex (8 chars), we just take the first 6
    if len(hex_str) == 8:
        hex_str = hex_str[:6]
    r, g, b = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)

def generate_interactive_map():
    sst_file = "sst_temp.nc"
    try:
        # 2. FIND LATEST DATE
        print("Finding latest available data date...")
        ds_meta = xr.open_dataset(BASE_URL, engine='netcdf4')
        latest_time = ds_meta.time.values[-1]
        time_str = np.datetime_as_string(latest_time, unit='s') + "Z"
        actual_date_only = str(latest_time)[:10]
        print(f"Latest data found for: {time_str}")

        # 3. DOWNLOAD
        download_url = f"{BASE_URL}.nc?sst[({time_str})][(0.0)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"
        print(f"Downloading slice: {download_url}")
        response = requests.get(download_url, timeout=120)
        response.raise_for_status()
        with open(sst_file, 'wb') as f:
            f.write(response.content)

        # 4. PROCESS
        ds_sst = xr.open_dataset(sst_file)
        # Handle potential empty dimensions and fill NaNs for the colormap math
        sst_vals = ds_sst.sst.squeeze().values
        # Replace NaNs with the mean so the colormap doesn't break
        sst_filled = np.nan_to_num(sst_vals, nan=np.nanmean(sst_vals))
        
        lats, lons = ds_sst.latitude.values, ds_sst.longitude.values
        
        # PFZ Calculation
        sst_smoothed = gaussian_filter(sst_filled, sigma=1.2)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        threshold = np.nanpercentile(grad_mag, 90)
        pfz_mask = np.where(grad_mag > threshold, 1.0, 0.0)
        pfz_smooth = gaussian_filter(pfz_mask, sigma=1.5)

        # 5. INITIALIZE MAP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        # FIXED COLORMAP: Use StepColormap or Linear with explicit boundaries
        vmin, vmax = float(np.nanmin(sst_vals)), float(np.nanmax(sst_vals))
        # Ensure vmin and vmax are not identical to prevent sorting errors
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        sst_colormap = cm.LinearColormap(
            colors=['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000'],
            index=np.linspace(vmin, vmax, 6), # Explicitly sorted thresholds
            vmin=vmin, vmax=vmax,
            caption=f"SST (°C) - {actual_date_only}"
        )
        
        raster_layers.ImageOverlay(
            image=np.flipud(sst_vals),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            opacity=0.5,
            name="Sea Surface Temperature",
            colormap=lambda x: hex_to_rgba(sst_colormap(x), alpha=0.6)
        ).add_to(m)
        m.add_child(sst_colormap)

        # 6. PFZ Heatmap
        pfz_data = []
        for i in range(0, len(lats), 2):
            for j in range(0, len(lons), 2):
                if pfz_smooth[i, j] > 0.4:
                    pfz_data.append([lats[i], lons[j], float(pfz_smooth[i, j])])
        
        plugins.HeatMap(pfz_data, name="Fishing Hotspots (PFZ)", min_opacity=0.4, 
                        radius=12, blur=8, 
                        gradient={0.4: 'blue', 0.7: 'lime', 1: 'white'}).add_to(m)

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        
        m.save("outputs/index.html")
        print(f"SUCCESS: Interactive map created for {actual_date_only}")

    except Exception as e:
        print(f"Failed: {e}")
        raise e
    finally:
        if os.path.exists(sst_file):
            os.remove(sst_file)

if __name__ == "__main__":
    generate_interactive_map()
