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

def hex_to_rgba(hex_str, alpha=0.6):
    """Safely converts hex strings to float tuples (R, G, B, A)"""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 8: hex_str = hex_str[:6] # Clip alpha if present
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

        # 4. PROCESS DATA
        ds_sst = xr.open_dataset(sst_file)
        sst_vals = ds_sst.sst.squeeze().values
        lats, lons = ds_sst.latitude.values, ds_sst.longitude.values
        
        # Clean data for PFZ calculation: replace NaNs with mean to avoid gradient errors
        mean_sst = np.nanmean(sst_vals)
        sst_filled = np.nan_to_num(sst_vals, nan=mean_sst)
        
        # PFZ Calculation (Fronts)
        sst_smoothed = gaussian_filter(sst_filled, sigma=1.2)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        threshold = np.nanpercentile(grad_mag, 90)
        pfz_mask = np.where(grad_mag > threshold, 1.0, 0.0)
        pfz_smooth = gaussian_filter(pfz_mask, sigma=1.5)

        # 5. INITIALIZE MAP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        # 6. STABLE COLORMAP LOGIC
        # We manually define 6 steps to ensure 'branca' never sees an unsorted list
        vmin = floor_val = np.floor(np.nanmin(sst_vals))
        vmax = ceil_val = np.ceil(np.nanmax(sst_vals))
        
        # If range is zero (bad data), create dummy range
        if vmin >= vmax:
            vmin, vmax = 20, 30
            
        colors = ['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000']
        sst_colormap = cm.LinearColormap(
            colors=colors,
            vmin=vmin,
            vmax=vmax
        ).to_step(n=12) # Use more steps for smoothness but keeps it discrete/stable
        
        sst_colormap.caption = f"SST (°C) - {actual_date_only}"

        # 7. ADD SST LAYER
        # We use a lambda that handles NaNs by returning a transparent color
        def color_logic(x):
            if np.isnan(x): return (0,0,0,0) # Transparent for land/NaN
            return hex_to_rgba(sst_colormap(x))

        raster_layers.ImageOverlay(
            image=np.flipud(sst_vals),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            opacity=0.7,
            name="Sea Surface Temperature",
            colormap=color_logic
        ).add_to(m)
        
        m.add_child(sst_colormap)

        # 8. ADD PFZ HEATMAP
        pfz_data = []
        for i in range(0, len(lats), 2):
            for j in range(0, len(lons), 2):
                if pfz_smooth[i, j] > 0.35:
                    pfz_data.append([lats[i], lons[j], float(pfz_smooth[i, j])])
        
        plugins.HeatMap(pfz_data, name="Fishing Hotspots (PFZ)", 
                        min_opacity=0.4, radius=10, blur=8, 
                        gradient={0.4: 'blue', 0.6: 'lime', 1: 'white'}).add_to(m)

        # 9. EXTRAS
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
