import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import date

# -----------------------
# 1. SETTINGS
# -----------------------
TODAY = date.today().isoformat()

LAT_MIN, LAT_MAX = 18, 24
LON_MIN, LON_MAX = 88, 94

# Example ERDDAP datasets (replace later with CMEMS)
SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaa_oisst.nc"
CHL_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla1day.nc"

# -----------------------
# 2. LOAD DATA
# -----------------------
sst = xr.open_dataset(SST_URL).sel(
    time=TODAY,
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)

chl = xr.open_dataset(CHL_URL).sel(
    time=TODAY,
    latitude=slice(LAT_MIN, LAT_MAX),
    longitude=slice(LON_MIN, LON_MAX)
)

# Rename for consistency
chl = chl.rename({"latitude": "lat", "longitude": "lon"})

# -----------------------
# 3. GRADIENTS (fronts)
# -----------------------
deg2km = 111.0

dTdx = sst.sst.differentiate("lon") / deg2km
dTdy = sst.sst.differentiate("lat") / deg2km
sst_grad = np.sqrt(dTdx**2 + dTdy**2)

dCdx = chl.chlorophyll.differentiate("lon") / deg2km
dCdy = chl.chlorophyll.differentiate("lat") / deg2km
chl_grad = np.sqrt(dCdx**2 + dCdy**2)

# -----------------------
# 4. PFZ LOGIC
# -----------------------
pfz = (sst_grad > 0.5) & (chl_grad > chl_grad.quantile(0.75))

# -----------------------
# 5. PLOT
# -----------------------
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

pfz.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="hot",
    add_colorbar=False
)

ax.coastlines()
ax.set_title(f"Potential Fishing Zones – {TODAY}")

plt.savefig(f"outputs/pfz_{TODAY}.png", dpi=300, bbox_inches="tight")
plt.close()

print("PFZ map generated successfully")
