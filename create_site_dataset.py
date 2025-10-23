import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# ========= CONFIGURATION =========
WAVEFORM_DIR = "waveforms/"              # Folder with waveform files
FLATFILE_PATH = "NGAW2_flatfile.csv"     # Your CSV file
OUTPUT_SITE_CSV = "site_data.csv"        # Output per-site dataset
MAX_WORKERS = 8                          # Number of threads for parallel processing

# ========= SUPPRESS WARNINGS =========
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========= 1. Load metadata (Flatfile) =========
print("📂 Loading NGA-West2 flatfile...")
flat = pd.read_csv(FLATFILE_PATH, low_memory=False)

# Ensure required columns exist
required_cols = [
    'Record Sequence Number', 'EQID', 'Earthquake Magnitude',
    'Joyner-Boore Dist. (km)', 'Vs30 (m/s) selected for analysis',
    'Station Name', 'Station Latitude', 'Station Longitude'
]
missing_cols = [c for c in required_cols if c not in flat.columns]
if missing_cols:
    raise ValueError(f"Missing columns in flatfile: {missing_cols}")

# Keep and rename relevant columns
flat = flat[required_cols].rename(columns={
    'Record Sequence Number': 'RSN',
    'Earthquake Magnitude': 'magnitude',
    'Joyner-Boore Dist. (km)': 'distance_km',
    'Vs30 (m/s) selected for analysis': 'vs30_site1',
    'Station Name': 'Station',
    'Station Latitude': 'Latitude',
    'Station Longitude': 'Longitude'
})
flat['RSN'] = flat['RSN'].astype(int)

# ========= 2. Preload waveform files =========
print("🔍 Scanning waveform directory...")
all_files = os.listdir(WAVEFORM_DIR)
file_dict = {}  # {rsn: {ext: [paths]}}

for f in all_files:
    match = re.match(r"RSN0*(\d+).*\.([AVD]T2)$", f, re.IGNORECASE)
    if match:
        rsn_num, ext = int(match.group(1)), match.group(2).upper()
        file_dict.setdefault(rsn_num, {}).setdefault(ext, []).append(os.path.join(WAVEFORM_DIR, f))

available_rsns = sorted(file_dict.keys())
print(f"✅ Found {len(available_rsns)} RSNs with available waveform files.")

# Filter the flatfile to only those RSNs that have waveforms
flat = flat[flat["RSN"].isin(available_rsns)].reset_index(drop=True)
print(f"📊 Processing {len(flat)} RSNs across {flat['EQID'].nunique()} earthquakes.\n")

# ========= 3. Waveform reading utility =========
def read_waveform(file_path):
    try:
        data = np.loadtxt(file_path, skiprows=4)
        if data.ndim > 1:
            return data[:, 1]  # use second column (acceleration, velocity, displacement)
        return data
    except Exception:
        return None

# ========= 4. Compute site features =========
def compute_site_features(rsn):
    results = {}
    files = file_dict.get(rsn, {})

    # --- Acceleration files (.AT2) ---
    for idx, path in enumerate(files.get('AT2', [])[:2]):
        acc = read_waveform(path)
        if acc is not None and len(acc) > 0:
            results[f"PGA_{idx}"] = np.max(np.abs(acc)) / 981.0  # convert to g

    # --- Velocity files (.VT2) ---
    for idx, path in enumerate(files.get('VT2', [])[:2]):
        vel = read_waveform(path)
        if vel is not None and len(vel) > 0:
            results[f"PGV_{idx}"] = np.max(np.abs(vel))

    # --- Displacement files (.DT2) ---
    for idx, path in enumerate(files.get('DT2', [])[:2]):
        disp = read_waveform(path)
        if disp is not None and len(disp) > 0:
            results[f"PGD_{idx}"] = np.max(np.abs(disp))

    # --- Compute averages ---
    pga_vals = [v for k, v in results.items() if k.startswith('PGA')]
    pgv_vals = [v for k, v in results.items() if k.startswith('PGV')]
    pgd_vals = [v for k, v in results.items() if k.startswith('PGD')]

    results['avg_pga'] = np.mean(pga_vals) if pga_vals else np.nan
    results['avg_pgv'] = np.mean(pgv_vals) if pgv_vals else np.nan
    results['avg_pgd'] = np.mean(pgd_vals) if pgd_vals else np.nan

    # --- Compute directionality θ ---
    if len(pga_vals) >= 2:
        results['delta_theta'] = np.degrees(np.arctan2(pga_vals[1], pga_vals[0]))
    else:
        results['delta_theta'] = np.nan

    return results

# ========= 5. Parallel processing =========
records = []

def process_row(row):
    rsn = int(row['RSN'])
    feats = compute_site_features(rsn)
    if feats:
        return {**row.to_dict(), **feats}
    return None

print("⚙️ Extracting site features (parallelized)...\n")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_row, row): idx for idx, row in flat.iterrows()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing RSNs"):
        res = future.result()
        if res:
            records.append(res)

# ========= 6. Save output =========
df_sites = pd.DataFrame(records)
df_sites.dropna(subset=['delta_theta'], inplace=True)
df_sites['delta_theta'] = np.clip(df_sites['delta_theta'], 0, 90)

df_sites.to_csv(OUTPUT_SITE_CSV, index=False)
print(f"\n✅ Created '{OUTPUT_SITE_CSV}' with {len(df_sites)} site entries from {df_sites['EQID'].nunique()} earthquakes.")
