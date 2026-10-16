from pathlib import Path
import pandas as pd

folder = Path("data")

for file in folder.iterdir():
    if file.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
        continue

    try:
        if file.suffix.lower() == ".csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        print("\n" + "=" * 50)
        print("FILE:", file.name)
        print("SHAPE:", df.shape)
        print("COLUMNS:", df.columns.tolist())
        print(df.head(3).to_string(index=False))

    except Exception as error:
        print("ERROR reading", file.name, ":", error)
