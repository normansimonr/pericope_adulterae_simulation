import csv
import io

data = [{"run_id": 0, "majority_text": "00000000", "pct": 0.5, "regime": "insertion"}]

output = io.StringIO()
fieldnames = ["run_id", "majority_text", "pct", "regime"]
writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
writer.writeheader()
writer.writerows(data)

print("--- Data as written ---")
print(output.getvalue())

# Now read it back
input_data = io.StringIO(output.getvalue())
reader = csv.DictReader(input_data)
row = next(reader)
print("--- Read back ---")
print(row)
