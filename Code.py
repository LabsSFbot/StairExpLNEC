import os
import pandas as pd
import numpy as np

# Parameters
fps = 25
dt = 1 / fps
input_dir = r"E:\comp\2M\2C\avi\segment final"
output_dir = os.path.join(input_dir, "resultatsIA4")
os.makedirs(output_dir, exist_ok=True)

# List to store all results
all_results = []

# List of CSV files to process
csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]

def calc_LS(x, y):
    """Calculate the horizontal segment length (LS)."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    if len(x_clean) < 3:
        return 0.0
    dx = np.diff(x_clean)

    leftward_indices = np.where(dx < 0)[0]
    if len(leftward_indices) == 0:
        return 0.0

    start_idx = leftward_indices[0]
    end_idx = None

    for i in range(start_idx, len(dx) - 1):
        if dx[i] > 0 and dx[i + 1] > 0:
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(x_clean) - 1

    if end_idx <= start_idx:
        end_idx = len(x_clean) - 1

    segment_x = x_clean[start_idx:end_idx + 1]
    LS = np.sum(np.abs(np.diff(segment_x)))
    return LS

def calc_trajectory_length(x, y):
    """Calculate total trajectory length (L) and horizontal segment length (LS)."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    dx, dy = np.diff(x[mask]), np.diff(y[mask])
    L = np.sum(np.sqrt(dx**2 + dy**2))
    LS = calc_LS(x, y)
    return L, LS

def trapezoidal_integration(y, x):
    """Manual trapezoidal integration."""
    return np.sum((y[:-1] + y[1:]) * np.diff(x) / 2)

def calculate_njs(jerk, time, duration, length_ratio):
    """Calculate Normalized Jerk Score."""
    integral_jerk_squared = trapezoidal_integration(jerk**2, time)
    njs = np.sqrt(0.5 * integral_jerk_squared * (duration**5) / (length_ratio**2))
    return njs

# Process each CSV file
for filename in csv_files:
    print(f"\nProcessing {filename}...")
    file_path = os.path.join(input_dir, filename)

    try:
        # Load the file
        df = pd.read_csv(file_path, header=None)
        header = df.iloc[:3]
        df = df.iloc[3:].copy().reset_index(drop=True)

        # Create column names
        columns = []
        for i in range(df.shape[1]):
            part = str(header.iloc[1, i]).strip() if len(header) > 1 else ""
            coord = str(header.iloc[2, i]).strip() if len(header) > 2 else ""
            if not part or part.lower() in ["nan", ""] or not coord or coord.lower() in ["nan", ""]:
                columns.append(f"col_{i}")
            else:
                columns.append(f"{part}_{coord}")
        df.columns = columns

        # Convert to float
        df = df.apply(pd.to_numeric, errors='coerce')

        # Clean data based on likelihood threshold
        bodyparts = ["Paw", "Helbow", "Shoulder"]
        threshold = 0.90
        conditions = pd.Series(True, index=df.index)
        for bp in bodyparts:
            conditions &= (df[f"{bp}_likelihood"].gt(threshold) & df[f"{bp}_x"].notna() & df[f"{bp}_y"].notna())
        df_clean = df[conditions].copy().reset_index(drop=True)

        # Calculate elbow angle
        df_clean['Helbow_angle'] = np.degrees(
            np.arctan2(df_clean['Paw_y'] - df_clean['Helbow_y'], df_clean['Paw_x'] - df_clean['Helbow_x']) -
            np.arctan2(df_clean['Shoulder_y'] - df_clean['Helbow_y'], df_clean['Shoulder_x'] - df_clean['Helbow_x'])
        )

        # Calculate velocity, acceleration, and jerk for Paw
        for bp in bodyparts:
            df_clean[f"{bp}_vx"] = df_clean[f"{bp}_x"].diff() / dt
            df_clean[f"{bp}_vy"] = df_clean[f"{bp}_y"].diff() / dt
            df_clean[f"{bp}_ax"] = df_clean[f"{bp}_vx"].diff() / dt
            df_clean[f"{bp}_ay"] = df_clean[f"{bp}_vy"].diff() / dt
            df_clean[f"{bp}_jerk"] = np.sqrt(
                (df_clean[f"{bp}_ax"].diff() / dt)**2 +
                (df_clean[f"{bp}_ay"].diff() / dt)**2
            )

        # Synergy analysis
        delta_paw_y = df_clean["Paw_y"].diff()
        delta_angle = df_clean["Helbow_angle"].diff()
        synergy = (np.sign(delta_paw_y) == np.sign(delta_angle)).astype(int)
        synergy_percent = 100 * synergy.sum() / len(synergy.dropna()) if len(synergy.dropna()) > 0 else np.nan

        # Phase-specific synergy
        df_clean["delta_paw_x"] = df_clean["Paw_x"].diff()
        df_clean["delta_paw_y"] = df_clean["Paw_y"].diff()
        df_clean["delta_angle"] = df_clean["Helbow_angle"].diff()

        advance_idx = df_clean[df_clean["delta_paw_x"] > 0].index
        lift_idx = df_clean[df_clean["delta_paw_y"] < 0].index
        drop_idx = df_clean[df_clean["delta_paw_y"] > 0].index

        def compute_synergy(indexes):
            valid = indexes.intersection(df_clean.dropna(subset=["delta_paw_y", "delta_angle"]).index)
            if len(valid) > 0:
                same_dir = (np.sign(df_clean.loc[valid, "delta_paw_y"]) == np.sign(df_clean.loc[valid, "delta_angle"])).sum()
                return 100 * same_dir / len(valid)
            else:
                return np.nan

        synergy_advance = compute_synergy(advance_idx)
        synergy_lift = compute_synergy(lift_idx)
        synergy_drop = compute_synergy(drop_idx)

        # Calculate trajectory lengths for Paw
        L_paw, LS_paw = calc_trajectory_length(df_clean["Paw_x"], df_clean["Paw_y"])
        length_ratio_paw = L_paw / LS_paw if LS_paw > 0 else np.nan

        # Calculate NJS for Paw
        jerk_paw = df_clean["Paw_jerk"].dropna().values
        time_jerk_paw = np.arange(len(jerk_paw)) * dt
        duration_paw = len(jerk_paw) * dt
        njs_paw = calculate_njs(jerk_paw, time_jerk_paw, duration_paw, length_ratio_paw) if len(jerk_paw) > 0 and not np.isnan(length_ratio_paw) else np.nan

        # Store results
        file_results = [
            {"File": filename, "Metric": "Global Synergy", "Result": synergy_percent, "Unit": "%"},
            {"File": filename, "Metric": "Advance Synergy", "Result": synergy_advance, "Unit": "%"},
            {"File": filename, "Metric": "Lift Synergy", "Result": synergy_lift, "Unit": "%"},
            {"File": filename, "Metric": "Drop Synergy", "Result": synergy_drop, "Unit": "%"},
            {"File": filename, "Metric": "Paw NJS (normalized by L/LS)", "Result": njs_paw, "Unit": ""},
        ]
        all_results.extend(file_results)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save all results to Excel
if all_results:
    df_results = pd.DataFrame(all_results)
    df_results.to_excel(
        os.path.join(output_dir, "Results.xlsx"),
        index=False,
        sheet_name="Results"
    )
    print("All results have been saved to 'Results.xlsx'.")
else:
    print("No results to save.")