import os
import xarray as xr
import numpy as np
import folium
from folium import plugins, raster_layers
import branca.colormap as cm
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
import requests
import time

# 1. SETUP
os.makedirs("outputs", exist_ok=True)
LAT_MIN, LAT_MAX = 14, 25
LON_MIN, LON_MAX = 85.5, 94
target_dt = datetime.utcnow() - timedelta(days=4)
date_str = target_dt.strftime('%Y-%m-%d')

# We'll download .nc files directly via HTTPS instead of opening them as remote streams
SST_DOWNLOAD_URL = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc?sst%5B({date_str}T12:00:00Z)%5D%5B(0.0)%5D%5B({LAT_MIN}):({LAT_MAX})%5D%5B({LON_MIN}):({LON_MAX})%5D"

def download_file(url, local_filename):
    print(f"Downloading data from: {url}")
    for i in range(3): # Try 3 times
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(10)
    return False

def hex_to_rgba(hex_str, alpha=1.0):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)

def generate_interactive_map():
    sst_file = "sst_temp.nc"
    try:
        if not download_file(SST_DOWNLOAD_URL, sst_file):
            raise Exception("Could not download SST data after multiple attempts.")

        # 2. LOAD DATA
        ds_sst = xr.open_dataset(sst_file)
        actual_date = str(ds_sst.time.values[0])[:10]
        sst_vals = ds_sst.sst.squeeze().values
        lats, lons = ds_sst.latitude.values, ds_sst.longitude.values
        
        # 3. CALCULATE PFZ
        sst_smoothed = gaussian_filter(sst_vals, sigma=1.2)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        threshold = np.nanpercentile(grad_mag, 90)
        pfz_mask = np.where(grad_mag > threshold, 1.0, 0.0)
        pfz_smooth = gaussian_filter(pfz_mask, sigma=1.5)

        # 4. INITIALIZE MAP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        sst_colormap = cm.LinearColormap(
            colors=['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000'],
            vmin=np.nanmin(sst_vals), vmax=np.nanmax(sst_vals),
            caption=f"SST (°C) - {actual_date}"
        )
        
        raster_layers.ImageOverlay(
            image=np.flipud(sst_vals),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            opacity=0.5,
            name="Sea Surface Temperature",
            colormap=lambda x: hex_to_rgba(sst_colormap(x), alpha=0.6)
        ).add_to(m)
        m.add_child(sst_colormap)

        # 5. PFZ Heatmap
        pfz_data = []
        for i in range(0, len(lats), 2):
            for j in range(0, len(lons), 2):
                if pfz_smooth[i, j] > 0.4:
                    pfz_data.append([lats[i], lons[j], pfz_smooth[i, j]])
        
        plugins.HeatMap(pfz_data, name="Fishing Hotspots (PFZ)", min_opacity=0.4, 
                        radius=12, blur=8, 
                        gradient={0.4: 'blue', 0.7: 'lime', 1: 'white'}).add_to(m)

        # 6. EXTRAS & SAVE
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        m.save("outputs/index.html")
        print(f"SUCCESS: Interactive map created for {actual_date}")

    except Exception as e:
        print(f"Failed: {e}")
        raise e
    finally:
        if os.path.exists(sst_file):
            os.remove(sst_file) # Clean up temp file

if __name__ == "__main__":
    generate_interactive_map()
