"""
05A.CreateOceanBasinMask.py
Builds a 1°×1° Natural-Earth ocean-basin mask and saves it compressed.
Run only once; output used by 05A.
"""
from pathlib import Path
import numpy as np
import regionmask

DATA = Path(__file__).resolve().parent.parent / "data"
OUT  = DATA / "ocean_mask_1deg.npz"

if OUT.exists():
    print("mask already exists →", OUT)
    quit()

# pick newest Natural-Earth mask available
try:
    basins = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
except AttributeError:
    try:
        basins = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50
    except AttributeError:
        basins = regionmask.defined_regions.natural_earth.ocean_basins_50   # fallback

lats = np.arange(-89.5,  90.5, 1.0)    # cell-centre grid
lons = np.arange(-179.5, 180.5, 1.0)
# build the 2-D (lat × lon) basin mask ────────────────────────────
try:                                    # newer regionmask (≥0.11)
    mask = basins.mask(lat=lats, lon=lons).values.astype("int16")
except TypeError:                       # older regionmask (<0.11)
    mask = basins.mask(lons, lats).astype("int16")     # positional


np.savez_compressed(OUT, mask=mask, lat0=-89.5, lon0=-179.5)
print(" 1° ocean-basin mask written →", OUT)
