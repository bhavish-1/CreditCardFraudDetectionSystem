import pandas as pd
import os

# ---- CONFIG ----
original_csv = "/Users/bhavish/Desktop/CreditCardFraudDetectinSystem/data/raw/fraudTest.csv"          # original file
output_1 = "/Users/bhavish/Desktop/CreditCardFraudDetectinSystem/data/raw/temp_1.csv"       # first half
output_2 = "/Users/bhavish/Desktop/CreditCardFraudDetectinSystem/data/raw/temp_2.csv"       # second half
# ----------------

# Load CSV
df = pd.read_csv(original_csv)

# Compute split index
mid = len(df) // 2

# Split into two halves
df_part_1 = df.iloc[:mid]
df_part_2 = df.iloc[mid:]

# Save both parts
df_part_1.to_csv(output_1, index=False)
df_part_2.to_csv(output_2, index=False)

# Delete original file
os.remove(original_csv)

print("✅ CSV split successfully!")
print(f"Saved: {output_1} → rows = {len(df_part_1)}")
print(f"Saved: {output_2} → rows = {len(df_part_2)}")
print("🗑️ Original file deleted.")
