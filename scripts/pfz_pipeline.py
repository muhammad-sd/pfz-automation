import os
import xarray as xr
import numpy as np
import folium
from folium import plugins, raster_layers
import branca.colormap as cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator # For the high-res effect
from datetime import datetime
import requests
import time

# 1. SETUP
os.makedirs("outputs", exist_ok=True)
LAT_MIN, LAT_MAX = 14, 25
LON_MIN, LON_MAX = 85.5, 94
# Increase this factor to make it smoother (4 means 4x more pixels)
SMOOTH_FACTOR = 4 

DATASET_ID = "ncdcOisst21Agg_LonPM180"
BASE_URL = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{DATASET_ID}"

def hex_to_rgba(hex_str, alpha=0.7):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 8: hex_str = hex_str[:6]
    r, g, b = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)

def generate_interactive_map():
    sst_file = "sst_temp.nc"
    try:
        # 2. FETCH LATEST DATA
        ds_meta = xr.open_dataset(BASE_URL, engine='netcdf4')
        latest_time = ds_meta.time.values[-1]
        time_str = np.datetime_as_string(latest_time, unit='s') + "Z"
        actual_date = str(latest_time)[:10]

        download_url = f"{BASE_URL}.nc?sst[({time_str})][(0.0)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"
        response = requests.get(download_url, timeout=120)
        with open(sst_file, 'wb') as f: f.write(response.content)

        # 3. INTERPOLATION (The "Magic" Step)
        ds = xr.open_dataset(sst_file)
        raw_sst = ds.sst.squeeze().values
        raw_lats, raw_lons = ds.latitude.values, ds.longitude.values

        # Create a new, finer grid
        fine_lats = np.linspace(raw_lats.min(), raw_lats.max(), len(raw_lats) * SMOOTH_FACTOR)
        fine_lons = np.linspace(raw_lons.min(), raw_lons.max(), len(raw_lons) * SMOOTH_FACTOR)
        
        # Interpolate raw data onto fine grid
        interp = RegularGridInterpolator((raw_lats, raw_lons), raw_sst, bounds_error=False, fill_value=None)
        grid_lon, grid_lat = np.meshgrid(fine_lons, fine_lats)
        sst_vals = interp((grid_lat, grid_lon))

        # 4. PFZ CALCULATION
        # We calculate fronts on the high-res data for better shapes
        sst_filled = np.nan_to_num(sst_vals, nan=np.nanmean(sst_vals))
        sst_smoothed = gaussian_filter(sst_filled, sigma=2.0)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        threshold = np.nanpercentile(grad_mag, 92) # Top 8% of fronts
        pfz_mask = np.where(grad_mag > threshold, 1.0, 0.0)
        pfz_smooth = gaussian_filter(pfz_mask, sigma=2.0)

        # 5. INITIALIZE MAP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        # 6. STYLISH COLORMAP
        vmin, vmax = np.nanmin(sst_vals), np.nanmax(sst_vals)
        sst_colormap = cm.LinearColormap(
            colors=['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000'],
            vmin=vmin, vmax=vmax
        ).to_step(n=15)
        sst_colormap.caption = f"SST (°C) - {actual_date}"

        # 7. ADD LAYERS
        raster_layers.ImageOverlay(
            image=np.flipud(sst_vals),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            name="Smooth Temperature Gradient",
            colormap=lambda x: hex_to_rgba(sst_colormap(x))
        ).add_to(m)
        
        # Adding the PFZ Heatmap with higher density
        pfz_data = []
        for i in range(0, len(fine_lats), 3):
            for j in range(0, len(fine_lons), 3):
                if pfz_smooth[i, j] > 0.3:
                    pfz_data.append([fine_lats[i], fine_lons[j], float(pfz_smooth[i, j])])
        
        plugins.HeatMap(pfz_data, name="Fishing Zones (Fronts)", 
                        min_opacity=0.5, radius=8, blur=5, 
                        gradient={0.4: 'blue', 0.6: 'lime', 1: 'white'}).add_to(m)

        m.add_child(sst_colormap)
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        
        m.save("outputs/index.html")
        print(f"SUCCESS: High-resolution map created.")

    except Exception as e:
        print(f"Failed: {e}")
        raise e
    finally:
        if os.path.exists(sst_file): os.remove(sst_file)

if __name__ == "__main__":
    generate_interactive_map()
