import pandas as pd
import numpy as np

# ---------- CONFIG ---------------------------------------------------------
INPUT_CSV     = 'DA.csv'             # original 20 000-row file
OUTPUT_SAMPLE = 'random_200.csv'     # 200-row sample to keep
SAMPLE_SIZE   = 200
COL_COPY_1    = 'mm2_per_px'         # existing column to copy
COL_COPY_2    = 'dark_px'           # another existing column (optional)
COL_COPY_3    = 'dark_mm2'           # another existing column (optional)
RANDOM_SEED   = 42
# ---------------------------------------------------------------------------

# 1) Load data --------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# 2) Sanity checks ----------------------------------------------------------
required = ['image', 'label', COL_COPY_1, COL_COPY_2, COL_COPY_3]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available columns:\n{df.columns.tolist()}")

# 3) Randomly sample 200 rows ----------------------------------------------
subset = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

# 4) Save the 200-row slice -------------------------------------------------
subset[['image', 'label', COL_COPY_1, COL_COPY_2, COL_COPY_3]].to_csv(OUTPUT_SAMPLE, index=False)

# 5) Create four *_random columns in-memory (not written to disk) ----------
for base in ['image', 'label', COL_COPY_1, COL_COPY_2, COL_COPY_3]:
    df[f'{base}_random'] = np.nan

df.loc[
    subset.index,
    [f'{c}_random' for c in ['image', 'label', COL_COPY_1, COL_COPY_2, COL_COPY_3]]
] = subset[['image', 'label', COL_COPY_1, COL_COPY_2, COL_COPY_3]].values

# 6) Done -------------------------------------------------------------------
print(f"✓ {SAMPLE_SIZE} sampled rows saved to {OUTPUT_SAMPLE}")
print("✓ Four *_random columns added to the DataFrame in memory (file not modified)")
