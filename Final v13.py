import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import calendar


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


CSV_URL = 'Motor_Vehicle_Collisions_-_Crashes_20251212.csv'


print("Loading 2024 NYC collision data...")
df = pd.read_csv(CSV_URL)


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


COL_DATE = 'crash_date'
COL_TIME = 'crash_time'
COL_INJURED = 'number_of_persons_injured'
COL_KILLED = 'number_of_persons_killed'
COL_STREET_ON = 'on_street_name'
COL_STREET_CROSS = 'cross_street_name'
COL_BOROUGH = 'borough'


vehicle_columns = [c for c in df.columns if 'vehicle_type' in c]
if vehicle_columns:
    COL_VEHICLE_TYPE = vehicle_columns[0]
else:
    COL_VEHICLE_TYPE = None


for col in [COL_INJURED, COL_KILLED]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = 0  


if COL_DATE in df.columns and COL_TIME in df.columns:
    df['crash_datetime'] = pd.to_datetime(df[COL_DATE] + ' ' + df[COL_TIME], errors='coerce')
else:
    raise KeyError("Date or Time columns are missing from the CSV!")

df.dropna(subset=['crash_datetime'], inplace=True)
df.set_index('crash_datetime', inplace=True)
df_2024 = df.copy()
print(f"Data loaded successfully. Total records: {len(df_2024)}")


total_crashes = len(df_2024)
total_injured = int(df_2024[COL_INJURED].sum()) if COL_INJURED in df_2024 else 0
total_killed = int(df_2024[COL_KILLED].sum()) if COL_KILLED in df_2024 else 0


street_columns = [c for c in [COL_STREET_ON, COL_STREET_CROSS] if c in df_2024]
street_counts = pd.concat([df_2024[col].dropna() for col in street_columns]).value_counts()
top_streets = street_counts.head(5)


df_2024['month'] = df_2024.index.month
monthly_accidents = df_2024['month'].value_counts().sort_index()
all_months = range(1, 13)
monthly_accidents_full = monthly_accidents.reindex(all_months, fill_value=0)
peak_month_number = monthly_accidents.idxmax()
peak_month_name = calendar.month_name[int(peak_month_number)]
peak_month_count = monthly_accidents.max()


if COL_VEHICLE_TYPE and COL_VEHICLE_TYPE in df_2024:
    vehicle_type_counts = df_2024[COL_VEHICLE_TYPE].replace('', np.nan).dropna().value_counts()
else:
    vehicle_type_counts = pd.Series([], dtype='int')

if not vehicle_type_counts.empty:
    most_common_vehicle = vehicle_type_counts.idxmax()
    most_common_vehicle_count = vehicle_type_counts.max()
    top_3_vehicles = vehicle_type_counts.head(3)
else:
    most_common_vehicle = "Unknown"
    most_common_vehicle_count = 0
    top_3_vehicles = pd.Series([], dtype='int')


def show_menu():
    print("\n---------------------------------------")
    print(" NYC Collision Data Viewer (2024)")
    print("---------------------------------------")
    print("Choose what data you want to view:")
    print("1. Total collisions")
    print("2. Total persons injured")
    print("3. Total persons killed")
    print("4. Month with most accidents")
    print("5. Top 5 most dangerous streets")
    print("6. Most common vehicle type involved")
    print("7. Monthly accident trend (line plot)")
    print("8. Vehicle types ranking (line plot)")
    print("9. Top 3 most common vehicle types (numbers)")
    print("10. Top 3 most common vehicle types (bar chart)")
    print("11. Pakyu Jios")
    print("12. Exit")
    
    print("---------------------------------------")


while True:
    show_menu()
    choice = input("Enter your choice (1–11): ")

    if choice == "1":
        print(f"\nTotal collisions in 2024: {total_crashes}")

    elif choice == "2":
        print(f"\nTotal persons injured in 2024: {total_injured}")

    elif choice == "3":
        print(f"\nTotal persons killed in 2024: {total_killed}")

    elif choice == "4":
        print(f"\nPeak month: {peak_month_name} ({peak_month_count} accidents)")

    elif choice == "5":
        print("\nTop 5 most dangerous streets:")
        if not top_streets.empty:
            for i, (street, count) in enumerate(top_streets.items(), start=1):
                print(f"{i}. {street}: {count} accidents")
        else:
            print("No street data available.")

    elif choice == "6":
        print("\nMost common vehicle type involved in collisions:")
        print(f"{most_common_vehicle} ({most_common_vehicle_count} collisions)")

    elif choice == "7":
        print("\nGenerating monthly accident trend plot...")
        months = [calendar.month_name[m] for m in all_months]
        plt.figure(figsize=(8, 4))
        plt.plot(months, monthly_accidents_full.values, marker='o', linestyle='-', color='crimson')
        plt.title("Monthly Motor Vehicle Collisions in NYC (2024)", fontsize=12)
        plt.xlabel("Month", fontsize=10)
        plt.ylabel("Number of Accidents", fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    elif choice == "8":
        print("\nGenerating vehicle types ranking plot...")
        if vehicle_type_counts.empty:
            print("No vehicle type data available.")
            continue
        vehicle_types = vehicle_type_counts.index
        counts = vehicle_type_counts.values
        plt.figure(figsize=(10, 5))
        plt.plot(vehicle_types, counts, marker='o', linestyle='-', color='navy')
        plt.xticks(rotation=45, ha='right')
        plt.title("Vehicle Types Involved in NYC Collisions (2024)", fontsize=12)
        plt.xlabel("Vehicle Type", fontsize=10)
        plt.ylabel("Number of Collisions", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    elif choice == "9":
        print("\nTop 3 most common vehicle types (numbers):")
        if top_3_vehicles.empty:
            print("No vehicle type data available.")
        else:
            for i, (veh, count) in enumerate(top_3_vehicles.items(), start=1):
                print(f"{i}. {veh}: {count} collisions")

    elif choice == "10":
        print("\nTop 3 most common vehicle types (bar chart):")
        if top_3_vehicles.empty:
            print("No vehicle type data available.")
        else:
            plt.figure(figsize=(6, 4))
            plt.bar(top_3_vehicles.index, top_3_vehicles.values, color='orange')
            plt.title("Top 3 Vehicle Types in NYC Collisions (2024)", fontsize=12)
            plt.xlabel("Vehicle Type", fontsize=10)
            plt.ylabel("Number of Collisions", fontsize=10)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    elif choice == "11":
        print("Pakyu Jios!")
        

    elif choice == "12":
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice. Please select a number from 1–11.")
