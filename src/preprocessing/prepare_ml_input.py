import csv

def main():
    input_file = "data/processed/observations_valid.csv"
    
    # Task 1: Loading and Converting Data
    accepted_records = []
    rejected_count = 0
    numeric_cols = ['temperature', 'velocity', 'altitude', 'signal_strength']
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                for col in numeric_cols:
                    row[col] = float(row[col])
                if row['altitude'] < 0:
                    rejected_count += 1
                    continue
                accepted_records.append(row)
            except ValueError:
                rejected_count += 1

    print("\n=== ML Input Preparation: Loading and Conversion ===")
    print(f"Input file: {input_file}")
    print(f"Records loaded: {len(accepted_records) + rejected_count}")
    print(f"Records accepted: {len(accepted_records)}")
    print(f"Records rejected: {rejected_count}")

    # Task 2: Normalization (Min-Max)
    for col in numeric_cols:
        col_values = [r[col] for r in accepted_records]
        min_val, max_val = min(col_values), max(col_values)
        for r in accepted_records:
            if max_val > min_val:
                r[col] = (r[col] - min_val) / (max_val - min_val)
            else:
                r[col] = 0.0

    print("\n=== ML Input Preparation: Normalization ===")
    print("Normalization completed successfully.")
    print("All selected numerical features are in range [0,1].")

    # Task 3 & 4: Derived & Temporal Features
    for r in accepted_records:
        r['temperature_velocity_interaction'] = r['temperature'] * r['velocity']
        r['altitude_signal_ratio'] = r['altitude'] / (r['signal_strength'] + 0.0001)
        
        try:
            unified_timestamp = r['timestamp'].replace(' ', 'T')
            hour_str = unified_timestamp.split('T')[1].split(':')[0]
            r['hour_normalized'] = float(hour_str) / 24.0
        except IndexError:
            r['hour_normalized'] = 0.0

    print("\n=== ML Input Preparation: Derived Features ===")
    print("New features added:\ntemperature_velocity_interaction\naltitude_signal_ratio")
    print("\n=== ML Input Preparation: Temporal Features ===")
    print("New feature added:\nhour_normalized")

    # Task 5 & 6: Feature Selection and Target Labels
    final_features = numeric_cols + ['temperature_velocity_interaction', 'altitude_signal_ratio', 'hour_normalized']
    
    model_features = []
    model_labels = []
    
    for r in accepted_records:
        feat_dict = {k: r[k] for k in final_features}
        model_features.append(feat_dict)
        model_labels.append({'anomaly_flag': r['anomaly_flag']})

    with open("data/processed/model_features.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_features)
        writer.writeheader()
        writer.writerows(model_features)

    with open("data/processed/model_labels.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['anomaly_flag'])
        writer.writeheader()
        writer.writerows(model_labels)

    print("\n=== ML Input Preparation: Feature Selection ===")
    print("Selected features:")
    for f in final_features: print(f)
    print("\n=== ML Input Preparation: Saving Outputs ===")
    print("Saved file: data/processed/model_features.csv")
    print("Saved file: data/processed/model_labels.csv")
    print(f"Number of records: {len(model_features)}")
    print(f"Number of features: {len(final_features)}")

if __name__ == "__main__":
    main()