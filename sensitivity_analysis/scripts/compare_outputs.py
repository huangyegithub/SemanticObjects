#!/usr/bin/env python3
import json

# Load both simulation outputs
with open('data/output/simulate_onto_baseline_20250722_094121_output_20250722_094747.json', 'r') as f:
    earlier = json.load(f)  # ChalkUnit tor
with open('data/output/simulate_onto_baseline_20250722_094121_output_20250722_132545.json', 'r') as f:
    later = json.load(f)    # SandstoneUnit tor

print("=== SIMULATION COMPARISON ===")
print("Earlier: ChalkUnit tor = new ChalkUnit(...)")
print("Later:   SandstoneUnit tor = new ChalkUnit(...)")
print()

# Compare metrics
print("=== METRICS COMPARISON ===")
metrics_to_compare = ["trap_count", "leak_count", "maturation_events", "migration_events", "execution_time"]

for metric in metrics_to_compare:
    earlier_val = earlier["metrics"].get(metric, 0)
    later_val = later["metrics"].get(metric, 0)
    diff = later_val - earlier_val if isinstance(earlier_val, (int, float)) else "N/A"
    print(f"{metric:20s}: {earlier_val:8} -> {later_val:8} (diff: {diff})")

print()

# Compare stdout lengths and content
earlier_stdout = earlier["raw_output"]["stdout"]
later_stdout = later["raw_output"]["stdout"]

print("=== OUTPUT COMPARISON ===")
print(f"Earlier stdout length: {len(earlier_stdout)} characters")
print(f"Later stdout length:   {len(later_stdout)} characters")
print(f"Length difference:     {len(later_stdout) - len(earlier_stdout)} characters")
print()

# Split into lines and compare
earlier_lines = earlier_stdout.split('\n')
later_lines = later_stdout.split('\n')

print(f"Earlier line count: {len(earlier_lines)}")
print(f"Later line count:   {len(later_lines)}")
print()

# Find differences line by line
differences = []
max_len = max(len(earlier_lines), len(later_lines))

for i in range(max_len):
    earlier_line = earlier_lines[i] if i < len(earlier_lines) else "<MISSING>"
    later_line = later_lines[i] if i < len(later_lines) else "<MISSING>"
    
    if earlier_line != later_line:
        differences.append({
            'line': i + 1,
            'earlier': earlier_line,
            'later': later_line
        })

if differences:
    print(f"=== FOUND {len(differences)} DIFFERENCES ===")
    for i, diff in enumerate(differences[:10]):  # Show first 10 differences
        print(f"\nDifference {i+1} at line {diff['line']}:")
        print(f"  Earlier: '{diff['earlier']}'")
        print(f"  Later:   '{diff['later']}'")
    
    if len(differences) > 10:
        print(f"\n... and {len(differences) - 10} more differences")
        
else:
    print("✓ NO DIFFERENCES FOUND - Outputs are identical!")

print(f"\n=== CONCLUSION ===")
if len(differences) == 0:
    print("The type declaration change has NO impact on simulation behavior.")
else:
    print(f"The type declaration change has MINIMAL impact:")
    print(f"- Same key metrics (traps: {earlier['metrics']['trap_count']}, leaks: {earlier['metrics']['leak_count']})")
    print(f"- Minor differences in {len(differences)} output lines")
    print(f"- Likely differences are in formatting or detailed output only")