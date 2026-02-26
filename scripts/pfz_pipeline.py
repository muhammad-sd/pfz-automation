import os
import xarray as xr
import numpy as np
import folium
from folium import plugins, raster_layers
import branca.colormap as cm
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

# 1. SETUP
os.makedirs("outputs", exist_ok=True)
LAT_MIN, LAT_MAX = 14, 25
LON_MIN, LON_MAX = 85.5, 94

target_dt = datetime.utcnow() - timedelta(days=4)

# Updated Dataset URLs
SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"
CURRENTS_URL = "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily"

def hex_to_rgba(hex_str, alpha=1.0):
    """Helper to convert #RRGGBB to (r, g, b, a) floats for Folium"""
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)

def generate_interactive_map():
    try:
        # 2. FETCH SST DATA
        print("Fetching SST data...")
        ds_sst = xr.open_dataset(SST_URL, engine='netcdf4')
        subset_sst = ds_sst.sel(time=target_dt, method='nearest').sel(
            latitude=slice(LAT_MIN, LAT_MAX), 
            longitude=slice(LON_MIN, LON_MAX)
        ).load()
        
        actual_date = str(subset_sst.time.values)[:10]
        sst_vals = subset_sst.sst.squeeze().values
        lats, lons = subset_sst.latitude.values, subset_sst.longitude.values
        
        # 3. CALCULATE PFZ
        sst_smoothed = gaussian_filter(sst_vals, sigma=1.2)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        threshold = np.nanpercentile(grad_mag, 90)
        pfz_mask = np.where(grad_mag > threshold, 1.0, 0.0)
        pfz_smooth = gaussian_filter(pfz_mask, sigma=1.5)

        # 4. FETCH CURRENT VECTORS
        print("Fetching Current vectors...")
        try:
            ds_curr = xr.open_dataset(CURRENTS_URL, engine='netcdf4')
            subset_curr = ds_curr.sel(time=target_dt, method='nearest').sel(
                latitude=slice(LAT_MIN, LAT_MAX), 
                longitude=slice(LON_MIN, LON_MAX)
            ).load()
            u_curr, v_curr = subset_curr.u_current.squeeze().values, subset_curr.v_current.squeeze().values
        except Exception as e:
            print(f"Currents data unavailable: {e}. Skipping arrows.")
            u_curr, v_curr = None, None

        # 5. INITIALIZE MAP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        # SST Heatmap Layer
        sst_colormap = cm.LinearColormap(
            colors=['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000'],
            vmin=np.nanmin(sst_vals), vmax=np.nanmax(sst_vals),
            caption=f"SST (°C) - {actual_date}"
        )
        
        # Corrected Colormap logic for ImageOverlay
        raster_layers.ImageOverlay(
            image=np.flipud(sst_vals),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            opacity=0.5,
            name="Sea Surface Temperature",
            colormap=lambda x: hex_to_rgba(sst_colormap(x), alpha=0.6)
        ).add_to(m)
        m.add_child(sst_colormap)

        # 6. PFZ Heatmap (Fishing Zones)
        pfz_data = []
        for i in range(0, len(lats), 2):
            for j in range(0, len(lons), 2):
                if pfz_smooth[i, j] > 0.4:
                    pfz_data.append([lats[i], lons[j], pfz_smooth[i, j]])
        
        plugins.HeatMap(pfz_data, name="Fishing Hotspots (PFZ)", min_opacity=0.4, 
                        radius=12, blur=8, 
                        gradient={0.4: 'blue', 0.7: 'lime', 1: 'white'}).add_to(m)

        # 7. ADD CURRENT VECTORS (Arrows)
        if u_curr is not None:
            curr_group = folium.FeatureGroup(name="Ocean Currents")
            c_lats, c_lons = subset_curr.latitude.values, subset_curr.longitude.values
            step = 5
            for i in range(0, len(c_lats), step):
                for j in range(0, len(c_lons), step):
                    u, v = u_curr[i, j], v_curr[i, j]
                    if not np.isnan(u) and not np.isnan(v):
                        mag = np.sqrt(u**2 + v**2)
                        if mag > 0.05:
                            folium.PolyLine(
                                locations=[[c_lats[i], c_lons[j]], 
                                           [c_lats[i] + v*0.8, c_lons[j] + u*0.8]],
                                color='white', weight=1, opacity=0.3
                            ).add_to(curr_group)
            curr_group.add_to(m)

        # 8. EXTRAS & SAVE
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        m.save("outputs/index.html")
        print("SUCCESS: Interactive map created.")

    except Exception as e:
        print(f"Failed: {e}")
        raise e

if __name__ == "__main__":
    generate_interactive_map()
