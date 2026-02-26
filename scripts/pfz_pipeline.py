import os
import xarray as xr
import numpy as np
import folium
from folium import plugins
import branca.colormap as cm
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

# 1. SETUP
os.makedirs("outputs", exist_ok=True)
LAT_MIN, LAT_MAX = 14, 25
LON_MIN, LON_MAX = 85.5, 94

target_dt = datetime.utcnow() - timedelta(days=4)

# Data URLs
SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"
# OSCAR Surface Currents (standard for surface flow)
CURRENTS_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/oscar_currents_interim"

def generate_interactive_map():
    try:
        # 2. FETCH SST DATA
        ds_sst = xr.open_dataset(SST_URL, engine='netcdf4')
        subset_sst = ds_sst.sel(time=target_dt, method='nearest').sel(
            latitude=slice(LAT_MIN, LAT_MAX), 
            longitude=slice(LON_MIN, LON_MAX)
        ).load()
        
        actual_date = str(subset_sst.time.values)[:10]
        sst_vals = subset_sst.sst.squeeze().values
        
        # 3. CALCULATE PFZ (Using your contour logic)
        sst_smoothed = gaussian_filter(sst_vals, sigma=1.2)
        dy, dx = np.gradient(sst_smoothed)
        grad_mag = np.sqrt(dx**2 + dy**2)
        threshold = np.nanpercentile(grad_mag, 90)
        pfz_mask = np.where(grad_mag > threshold, 1.0, 0.0)
        pfz_smooth = gaussian_filter(pfz_mask, sigma=1.5)

        # 4. FETCH CURRENT VECTORS (Simplified Windy Effect)
        # Note: Currents often have lower resolution than SST
        try:
            ds_curr = xr.open_dataset(CURRENTS_URL, engine='netcdf4')
            subset_curr = ds_curr.sel(time=target_dt, method='nearest').sel(
                latitude=slice(LAT_MIN, LAT_MAX), 
                longitude=slice(LON_MIN, LON_MAX)
            ).load()
            u_curr = subset_curr.u.squeeze().values
            v_curr = subset_curr.v.squeeze().values
        except Exception as e:
            print(f"Currents data unavailable: {e}")
            u_curr, v_curr = None, None

        # 5. INITIALIZE FOLIUM MAP
        m = folium.Map(location=[19.5, 89.5], zoom_start=6, tiles='CartoDB dark_matter')

        # SST Heatmap Layer
        sst_colormap = cm.LinearColormap(
            colors=['#000044', '#0044ff', '#00ffff', '#ffff00', '#ff4400', '#ff0000'],
            vmin=np.nanmin(sst_vals), vmax=np.nanmax(sst_vals),
            caption=f"SST (°C) - {actual_date}"
        )
        
        # ImageOverlay for smooth SST transition
        plugins.ImageOverlay(
            image=np.flipud(sst_vals),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            opacity=0.5,
            name="Sea Surface Temperature",
            interactive=True
        ).add_to(m)
        m.add_child(sst_colormap)

        # PFZ Overlay (Contour-like blobs)
        # For simplicity in Folium, we use the smoothed mask as a localized heatmap
        pfz_data = []
        lats = subset_sst.latitude.values
        lons = subset_sst.longitude.values
        for i in range(0, len(lats), 2): # Stepping for performance
            for j in range(0, len(lons), 2):
                if pfz_smooth[i, j] > 0.4:
                    pfz_data.append([lats[i], lons[j], pfz_smooth[i, j]])

        plugins.HeatMap(pfz_data, name="Fishing Hotspots", min_opacity=0.5, 
                        radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 1: 'white'}).add_to(m)

        # 6. ADD CURRENT VECTORS (Arrows)
        if u_curr is not None:
            curr_group = folium.FeatureGroup(name="Surface Currents")
            # We sample the grid so the map isn't crowded
            step = 3 
            c_lats = subset_curr.latitude.values
            c_lons = subset_curr.longitude.values
            for i in range(0, len(c_lats), step):
                for j in range(0, len(c_lons), step):
                    u, v = u_curr[i, j], v_curr[i, j]
                    if not np.isnan(u) and not np.isnan(v):
                        mag = np.sqrt(u**2 + v**2)
                        if mag > 0.1: # Only draw moving water
                            folium.PolyLine(
                                locations=[[c_lats[i], c_lons[j]], 
                                           [c_lats[i] + v*0.5, c_lons[j] + u*0.5]],
                                color='white', weight=1, opacity=0.6
                            ).add_to(curr_group)
            curr_group.add_to(m)

        # 7. EXTRAS & SAVE
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        m.save("outputs/index.html")
        print("SUCCESS: Interactive map created.")

    except Exception as e:
        print(f"Failed: {e}")
        raise e

if __name__ == "__main__":
    generate_interactive_map()
