import pandas as pd
import numpy as np
from itertools import combinations
from geopy.distance import geodesic
from tqdm import tqdm
import os

# ========= CONFIGURATION =========
SITE_DATA_PATH = "site_data.csv"     # input from create_site_dataset_all_events.py
OUTPUT_PAIR_CSV = "pair_data.csv"    # output pairwise data
MAX_PAIR_DISTANCE_KM = 125           # "nearby" threshold (can tune based on use-case)

# ========= 1. Load site data =========
if not os.path.exists(SITE_DATA_PATH):
    raise FileNotFoundError(f"❌ Site data file '{SITE_DATA_PATH}' not found. Run create_site_dataset_all_events.py first.")

df = pd.read_csv(SITE_DATA_PATH)
print(f"📂 Loaded {len(df)} sites from {df['EQID'].nunique() if 'EQID' in df.columns else '?'} earthquakes.")

# ========= 2. Validate columns =========
required_cols = ['RSN', 'Latitude', 'Longitude', 'magnitude', 'vs30_site1', 'avg_pga', 'delta_theta']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in site_data.csv")

# If EQID missing (edge case), assign pseudo EQID
if "EQID" not in df.columns:
    print("⚠️  'EQID' column missing. Treating all sites as one event (EQID=0).")
    df["EQID"] = 0

# ========= 3. Generate pairs =========
pairs = []
eqid_groups = list(df.groupby("EQID"))

print(f"⚙️ Generating site pairs within each earthquake (max distance = {MAX_PAIR_DISTANCE_KM} km)...")

for eqid, group in tqdm(eqid_groups, desc="Processing events"):
    sites = group.to_dict(orient="records")
    if len(sites) < 2:
        continue  # skip single-site events

    for s1, s2 in combinations(sites, 2):
        coord1 = (s1['Latitude'], s1['Longitude'])
        coord2 = (s2['Latitude'], s2['Longitude'])

        # Compute distance safely
        try:
            dist_km = geodesic(coord1, coord2).km
        except Exception:
            continue

        if dist_km <= MAX_PAIR_DISTANCE_KM:
            delta_theta = abs(s1['delta_theta'] - s2['delta_theta'])
            delta_vs30 = abs(s1['vs30_site1'] - s2['vs30_site1'])
            avg_pga = np.nanmean([s1['avg_pga'], s2['avg_pga']])

            pairs.append({
                "EQID": eqid,
                "RSN_1": s1['RSN'],
                "RSN_2": s2['RSN'],
                "distance_km": dist_km,
                "magnitude": s1['magnitude'],
                "vs30_1": s1['vs30_site1'],
                "vs30_2": s2['vs30_site1'],
                "delta_vs30": delta_vs30,
                "avg_pga_pair": avg_pga,
                "delta_theta_target": delta_theta
            })

# ========= 4. Save output =========
df_pairs = pd.DataFrame(pairs)

if len(df_pairs) == 0:
    print("⚠️ No valid site pairs found. Check if MAX_PAIR_DISTANCE_KM is too small or site data is incomplete.")
else:
    df_pairs.to_csv(OUTPUT_PAIR_CSV, index=False)
    print(f"\n✅ Created '{OUTPUT_PAIR_CSV}' with {len(df_pairs)} site pairs from {df_pairs['EQID'].nunique()} earthquakes.")
