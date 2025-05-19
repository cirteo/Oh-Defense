import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read JSON file
def read_attention_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Convert data to matrix format
def process_attention_data(data):
    # Create a 32x32 zero matrix
    matrix = np.zeros((32, 32))
    
    # Fill in the data
    for key, value in data.items():
        layer, head = map(int, key.split('-'))
        # Swap layer and head positions to change matrix orientation
        matrix[head][layer] = value
        
    return matrix

# Draw heatmap
def plot_attention_heatmap(matrix, save_path=None):
    plt.figure(figsize=(12, 10))

    sns.heatmap(matrix,
                cmap='Oranges',  # Use orange gradient color spectrum
                xticklabels=range(32),
                yticklabels=range(32),
                cbar_kws={'label': 'Deflection Angle'},
                vmin=0,
                #vmax=30
                )  # Set maximum value to 50
    
    #plt.title('Shifts')
    plt.xlabel('Layer')  # Change to Layer
    plt.ylabel('Head')   # Change to Head
    
    # Adjust layout
    plt.tight_layout()
    
    # If save path is specified, save the image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def main():
    # Read and process data
    data = read_attention_data('angle/llava-v1.5-7b/safe.json')
    attention_matrix = process_attention_data(data)
    
    # Draw heatmap
    plot_attention_heatmap(attention_matrix, 'angle/llava-v1.5-7b/safe.png')
    
if __name__ == "__main__":
    main()