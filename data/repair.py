import pandas as pd

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]

def get_valid_input(col_name):
    while True:
        value = input(f"Enter corrected value for {col_name}: ").strip()

        try:
            if col_name == "datetime":
                return pd.to_datetime(value)
            return float(value)
        except Exception:
            print("Invalid input. Please try again.")

def repair_bad_rows(original_bad_df: pd.DataFrame, parsed_bad_df: pd.DataFrame) -> pd.DataFrame:
    if original_bad_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    repaired_rows = []

    for i in range(len(original_bad_df)):
        original_row = original_bad_df.iloc[i].copy()
        parsed_row = parsed_bad_df.iloc[i]

        print(f"\nFix row {i + 1}:")
        print(original_row.to_dict())

        for col in REQUIRED_COLUMNS:
            needs_repair = False

            if col in parsed_row.index and pd.isna(parsed_row[col]):
                needs_repair = True

            elif col != "datetime" and col in parsed_row.index:
                value = parsed_row[col]
                if isinstance(value, (int, float)) and value == 0:
                    needs_repair = True

            if needs_repair:
                original_row[col] = get_valid_input(col)

        repaired_rows.append(original_row)

    repaired_df = pd.DataFrame(repaired_rows, columns=REQUIRED_COLUMNS)

    repaired_df["datetime"] = pd.to_datetime(repaired_df["datetime"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        repaired_df[col] = pd.to_numeric(repaired_df[col], errors="coerce")

    repaired_df = repaired_df.dropna(subset=REQUIRED_COLUMNS)
    repaired_df = repaired_df.sort_values("datetime").reset_index(drop=True)

    return repaired_df