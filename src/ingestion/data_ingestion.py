import json
import csv
import os

def main():
    dataset_path = "data/raw/orbital_observations.csv"
    metadata_path = "data/raw/metadata.json"

    # Task 4: Load Metadata and Dataset
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    rows = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        dataset_columns = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Dataset: {metadata.get('dataset_name', 'orbital_observations')}")
    print(f"Records loaded: {len(rows)}")
    print(f"Columns (dataset): {dataset_columns}")
    metadata_columns = metadata.get("columns", [])
    print(f"Columns (metadata): {metadata_columns}")

    # Task 5: Column Consistency Validation
    if dataset_columns == metadata_columns:
        print("Column validation: OK")
        col_val_result = "OK"
    else:
        print("Column validation: MISMATCH")
        print(f"Expected: {metadata_columns}")
        print(f"Actual: {dataset_columns}")
        col_val_result = "MISMATCH"

    # Task 6: Record Count Validation
    expected_records = metadata.get("num_records", 0)
    if len(rows) == expected_records:
        print("Record count validation: OK")
        rec_val_result = "OK"
    else:
        print("Record count validation: MISMATCH")
        print(f"Expected: {expected_records}")
        print(f"Actual: {len(rows)}")
        rec_val_result = "MISMATCH"

    # Task 7: Detecting Invalid Records
    valid_records = []
    invalid_records = []
    for row in rows:
        # Zakładamy, że sprawdzamy np. kolumnę temperature pod kątem napisu "INVALID"
        if "INVALID" in row.values() or row.get("temperature") == "INVALID":
            invalid_records.append(row)
        else:
            valid_records.append(row)

    print(f"Valid: {len(valid_records)}")
    print(f"Invalid: {len(invalid_records)}")

    # Task 8: Saving Processed Outputs
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    with open("data/processed/observations_valid.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dataset_columns)
        writer.writeheader()
        writer.writerows(valid_records)

    with open("data/processed/observations_invalid.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dataset_columns)
        writer.writeheader()
        writer.writerows(invalid_records)

    # Task 9: Preparing Data for Preprocessing and Model Input
    feature_columns = metadata.get("feature_columns", [])
    with open("data/processed/model_input.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=feature_columns)
        writer.writeheader()
        for row in valid_records:
            writer.writerow({k: row[k] for k in feature_columns if k in row})

    # Task 10: Ingestion Summary
    summary = (
        f"Dataset: {metadata.get('dataset_name', 'orbital_observations')}\n"
        f"Records loaded: {len(rows)}\n"
        f"Expected records: {expected_records}\n"
        f"Column validation: {col_val_result}\n"
        f"Record count validation: {rec_val_result}\n"
        f"Valid records: {len(valid_records)}\n"
        f"Invalid records: {len(invalid_records)}\n"
        f"Generated files:\n"
        f"  data/processed/observations_valid.csv\n"
        f"  data/processed/observations_invalid.csv\n"
        f"  data/processed/model_input.csv\n"
    )
    with open("reports/ingestion_summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main()