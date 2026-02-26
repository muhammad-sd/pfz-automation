import os
import xarray as xr
import numpy as np
import folium
from folium import plugins, raster_layers
import branca.colormap as cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import requests

# 1. SETUP
os.makedirs("outputs", exist_ok=True)
LAT_MIN, LAT_MAX = 14, 25
LON_MIN, LON_MAX = 85.5, 94
SMOOTH_FACTOR = 4 # Upsamples the 25km data to ~6km for a smoother look

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
        # 2. GET DATA
        ds_meta = xr.open_dataset(BASE_URL, engine='netcdf4')
        latest_time = ds_meta.time.values[-1]
        time_str = np.datetime_as_string(latest_time, unit='s') + "Z"
        
        download_url = f"{BASE_URL}.nc?sst[({time_str})][(0.0)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"
        response = requests.get(download_url, timeout=120)
        with open(sst_file, 'wb') as f: f.write(response.content)

        # 3. ADVANCED PROCESSING
        ds = xr.open_dataset(sst_file)
        raw_sst = ds.sst.squeeze().values
        raw_lats, raw_lons = ds.latitude.values, ds.longitude.values

        # INTERPOLATION: This removes the "blocky" pixels
        fine_lats = np.linspace(raw_lats.min(), raw_lats.max(), len(raw_lats) * SMOOTH_FACTOR)
        fine_lons = np.linspace(raw_lons.min(), raw_lons.max(), len(raw_lons) * SMOOTH_FACTOR)
        interp = RegularGridInterpolator((raw_lats, raw_lons), raw_sst, bounds_error=False, fill_value=np.nan)
        grid_lon, grid_lat = np.meshgrid(fine_lons, fine_lats)
        sst_vals = interp((grid_lat, grid_lon))

        # PFZ LOGIC: Run on a filled version so gradients don't break at the coast
        sst_filled = np.nan_to_num(sst_vals, nan=np.nanmean(sst_vals))
        sst_smoothed = gaussian_filter(sst_filled, sigma=2.0)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        
        # MASKING: Ensure the Land stays transparent (Crucial for fixing the "Red Block")
        # We keep the original NaNs from the dataset to mask the overlay
        sst_masked = np.where(np.isnan(sst_vals), np.nan, sst_vals)

        # 4. MAP SETUP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        vmin, vmax = np.nanmin(raw_sst), np.nanmax(raw_sst)
        sst_colormap = cm.LinearColormap(
            colors=['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000'],
            vmin=vmin, vmax=vmax
        ).to_step(n=15)

        # 5. LAYERS
        # This function handles the transparency for land
        def color_logic(x):
            if np.isnan(x): return (0, 0, 0, 0) # Zero alpha for land
            return hex_to_rgba(sst_colormap(x))

        raster_layers.ImageOverlay(
            image=np.flipud(sst_masked),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            name="Smooth Temperature",
            colormap=color_logic
        ).add_to(m)

        # ADD PFZ (Only where gradient is high)
        pfz_points = []
        threshold = np.nanpercentile(grad_mag, 92)
        for i in range(0, len(fine_lats), 4):
            for j in range(0, len(fine_lons), 4):
                if grad_mag[i,j] > threshold and not np.isnan(sst_vals[i,j]):
                    pfz_points.append([fine_lats[i], fine_lons[j]])

        for pt in pfz_points:
            folium.CircleMarker(pt, radius=1, color='#00ff00', fill=True).add_to(m)

        m.add_child(sst_colormap)
        folium.LayerControl().add_to(m)
        m.save("outputs/index.html")
        print("SUCCESS: High-quality map saved.")

    except Exception as e:
        print(f"Failed: {e}")
    finally:
        if os.path.exists(sst_file): os.remove(sst_file)

if __name__ == "__main__":
    generate_interactive_map()
