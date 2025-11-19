import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import json
import numpy as np

def find_safe_heads(unsafe_file, safe_file, percentile=80, top_k=None):
    # Read two JSON files
    with open(unsafe_file, 'r') as f1, open(safe_file, 'r') as f2:
        unsafe_data = json.load(f1)
        safe_data = json.load(f2)
    
    # Calculate percentile value from safe file as threshold
    safe_values = list(safe_data.values())
    threshold = np.percentile(safe_values, percentile)
    print(f"{percentile}th percentile threshold: {threshold:.4f}")

    # Find heads with value below threshold in safe data
    filtered_heads = {}
    for key in unsafe_data:
        if key in safe_data:
            # Condition: value in safe data is below threshold
            if safe_data[key] < threshold:
                # Calculate difference (unsafe - safe)
                diff = unsafe_data[key] - safe_data[key]
                filtered_heads[key] = diff
    
    # Sort by difference (descending)
    sorted_heads = sorted(filtered_heads.items(), key=lambda x: x[1], reverse=True)
    
    # Return top k results if specified
    if top_k is not None and top_k < len(sorted_heads):
        sorted_heads = sorted_heads[:top_k]
    
    return sorted_heads, unsafe_data, safe_data  # Also return full data for reference

# Example usage
unsafe_file = '../angle/llava-v1.5-7b/vlsafe.json'    # Unsafe dataset
safe_file = '../angle/llava-v1.5-7b/safe.json'        # Safe dataset
percentile = 80  # Use 80th percentile as threshold
top_k = 10       # Return top 10 attention heads

result, unsafe_data, safe_data = find_safe_heads(unsafe_file, safe_file, percentile, top_k)

print(f"Attention heads (matching condition: safe value < {percentile}th percentile threshold):")
print(f"Found {len(result)} matching attention heads (showing top {top_k if top_k and top_k < len(result) else len(result)})")
for key, diff in result:
    print(f"{key}: {diff:.4f} (unsafe: {unsafe_data[key]:.4f}, safe: {safe_data[key]:.4f})")