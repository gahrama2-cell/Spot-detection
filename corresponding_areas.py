import pandas as pd
from pathlib import Path

# ---------- CONFIGURATION ---------------------------------
SOURCE_FILE      = Path("speck_report_mm2_with_cropped_images_pyrMeanShiftFiltering.csv")            # path to your original file
DEST_FILE        = Path("interestedlabel_areas_pyrMeanShiftFiltering.csv")      # path for the new file
LABEL_COL        = "label"                  # column that holds label names
AREA_COL         = "dark_mm2"                       # column that holds areas
LABELS_OF_INTEREST = [
    "r1774", "r1527", "r14129", "r947", "r8430", "r14996", "r15627", "r8497",
    "18584", "r14951", "r5349", "r10650", "r8093", "r2018", "r767", "r14719",
    "r13643", "18094", "r7272", "r718", "r15239", "r15752", "17764", "r13039",
    "r11645", "r5229", "r6770", "r7087", "r16832", "r8064", "r2458", "r12260",
    "r11350", "r5594", "r8425", "r12314", "r12724", "r3890", "r14738", "r5510",
    "r6046", "r5157", "r3746", "r8504", "r7364", "r3844", "r8334", "r13087",
    "r10207", "r4717", "r1386", "r5620", "r5154", "r15853", "r6631", "r1183",
    "r14010", "r897", "r13303", "r6906", "r11293", "r471", "r4003", "r15578",
    "r16031", "r9515", "r8128", "Spot609", "r17125", "r14095", "r14383", "r5389",
    "r11078", "r1413", "r13408", "r14461", "r264", "17550", "r5534", "r7607",
    "r5489", "r2637", "r8673", "r295", "r3808", "17875", "r10621", "r93",
    "r7066", "r10707", "r119", "r14101", "r14521", "Spot201", "r4594", "r13633",
    "r1650", "r6567", "r11216", "r325", "r11312", "r161", "r16281", "r15951",
    "r5819", "r16327", "r14028", "r9963", "r1571", "r533", "r2779", "r4352",
    "r11426", "r9617", "r11342", "r7652", "Spot768", "r6599", "r3237", "r11383",
    "r9827", "r15676", "r3441", "r2128", "r3146", "Spot589", "r10644", "r6582",
    "r363", "r15387", "r5957", "r7843", "r4271", "r14024", "r925", "r17122",
    "18102", "r9303", "r2028", "r617", "r16717", "r6470", "r14921", "r12494",
    "r631", "r12694", "r8784", "Spot219", "r1422", "18249", "r6905", "r16043",
    "r3263", "r1886", "r15042", "r1171", "r15438", "r3347", "Spot324", "Spot754",
    "r8118", "r14812", "r11542", "r3718", "r14285", "r214", "r5282", "r8644",
    "r4519", "r10917", "r4771", "r13581", "r9345", "r5884", "r334", "Spot715",
    "r3601", "r5072", "r12908", "r10770", "r7722", "r4327", "r5622", "r8004",
    "r7570", "r12493", "r6693", "r15867", "r6835", "r9915", "r3700", "r13356",
    "r16114", "r7327", "r10684", "r7184", "r4628", "r10337", "r15698", "r14453",
]
# ----------------------------------------------------------

def main():
    # 1. Load only the two columns we need
    df = pd.read_csv(SOURCE_FILE, usecols=[LABEL_COL, AREA_COL])

    # 2. Keep just the labels we care about
    df = df[df[LABEL_COL].isin(LABELS_OF_INTEREST)]

    # 3. Aggregate (delete this block if you prefer raw rows)
    df = (df
          .groupby(LABEL_COL, as_index=False)[AREA_COL]
          .sum())

    # 4. Re-order rows to match LABELS_OF_INTEREST exactly
    df = (df
          .set_index(LABEL_COL)                       # index by label
          .reindex(LABELS_OF_INTEREST)    # re-order (keeps NaN if a label is missing)
          .reset_index())

    # 5. Write the result
    df.to_csv(DEST_FILE, index=False)
    print(f"✓  Wrote {len(df)} rows to {DEST_FILE}")

if __name__ == "__main__":
    main()

