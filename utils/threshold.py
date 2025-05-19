import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import argparse

# Load data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Create distribution plot
def plot_distribution(df1, df2, file1_name, file2_name):
    # Create a single figure
    fig, ax = plt.subplots(figsize=(12, 6))
    max_score = max(df1['Deflection_Angle'].max(), df2['Deflection_Angle'].max())
    
    # Define specific colors
    color1 = (173/255, 217/255, 238/255)  # Light blue
    color2 = (248/255, 203/255, 173/255)  # Light orange
    # Combined density plot to compare two distributions
    sns.kdeplot(df1['Deflection_Angle'], color=color1, label=file1_name, ax=ax, fill=True, alpha=0.7)
    sns.kdeplot(df2['Deflection_Angle'], color=color2, label=file2_name, ax=ax, fill=True, alpha=0.7)
    ax.set_xlim(0, max_score)
    ax.set_title("Comparison of Deflection Angle's Distributions")
    ax.set_xlabel('Deflection Angle')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Find potential threshold points
    min_score = min(df1['Deflection_Angle'].min(), df2['Deflection_Angle'].min())
    max_score = max(df1['Deflection_Angle'].max(), df2['Deflection_Angle'].max())
    potential_thresholds = find_threshold_points(df1, df2, min_score, max_score)
    y_mid = ax.get_ylim()[1] / 2

    for threshold, score in potential_thresholds:
        ax.axvline(x=threshold, linestyle='--', color='red', alpha=1)

        ax.text(threshold + 0.01, y_mid, f'{threshold:.4f}', color='red', 
                rotation=0, horizontalalignment='left', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('threshold.png', dpi=300)
    plt.show()
    
    return potential_thresholds

# Fixed function to find threshold points
def find_threshold_points(df1, df2, min_score, max_score):
    # Create more detailed points to evaluate density
    points = np.linspace(min_score, max_score, 1000)
    
    # Use scipy's gaussian_kde to directly estimate density function
    kde1 = gaussian_kde(df1['Deflection_Angle'])
    kde2 = gaussian_kde(df2['Deflection_Angle'])
    
    # Calculate density values for each point
    densities1 = kde1(points)
    densities2 = kde2(points)
    density_diff = densities1 - densities2
    
    threshold_points = []
    for i in range(1, len(density_diff)):
        # If density difference sign changes, there is a crossing point
        if (density_diff[i-1] * density_diff[i]) <= 0 and density_diff[i-1] != 0 and density_diff[i] != 0:
            # Calculate crossing point position (linear interpolation)
            cross_point = points[i-1] + (points[i] - points[i-1]) * (0 - density_diff[i-1]) / (density_diff[i] - density_diff[i-1])
            threshold_points.append((cross_point, max(kde1(cross_point)[0], kde2(cross_point)[0])))
    
    # Evaluate separation effect of each threshold point
    thresholds_with_scores = []
    for threshold, _ in threshold_points:
        # Calculate number of misclassifications
        misclassified1 = len(df1[df1['Deflection_Angle'] < threshold]) / len(df1)
        misclassified2 = len(df2[df2['Deflection_Angle'] >= threshold]) / len(df2)
        score = 1 - (misclassified1 + misclassified2) / 2
        thresholds_with_scores.append((threshold, score))
    
    # If no crossing points found, manually add some potential thresholds
    if not thresholds_with_scores:
        mid_point = (df1['Deflection_Angle'].median() + df2['Deflection_Angle'].median()) / 2
        thresholds_with_scores.append((mid_point, 0.5))
    
    # Sort by separation effect and return top 3
    thresholds_with_scores.sort(key=lambda x: x[1], reverse=True)
    return thresholds_with_scores[:3]

# Analyze statistical differences between two files
def analyze_statistics(df1, df2, file1_name, file2_name):
    stats1 = df1['Deflection_Angle'].describe()
    stats2 = df2['Deflection_Angle'].describe()
    
    # Create comparative statistics table
    stats_df = pd.DataFrame({
        file1_name: stats1,
        file2_name: stats2
    })
    
    print("\n===== Deflection_Angle Statistical Analysis =====")
    print(stats_df)
    print("\n===== T-test Results =====")
    
    # Perform T-test
    t_stat, p_value = stats.ttest_ind(df1['Deflection_Angle'], df2['Deflection_Angle'], equal_var=False)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: The distributions are significantly different (p < 0.05)")
    else:
        print("Conclusion: The distributions are not significantly different (p >= 0.05)")
    
    return stats_df


# Main function
def main():
    parser = argparse.ArgumentParser(description='Compare two distributions and find threshold points.')
    
    parser.add_argument('--file1', type=str, required=True, 
                        help='Path to the first CSV file')
    parser.add_argument('--file2', type=str, required=True,
                        help='Path to the second CSV file')

    args = parser.parse_args()
    file1_path = args.file1
    file2_path = args.file2
    file1_name = file1_path.split('/')[-2] if '/' in file1_path else file1_path
    file2_name = file2_path.split('/')[-2] if '/' in file2_path else file2_path


    # Load data
    print("Loading data...")
    df1 = load_data(file1_path)
    df2 = load_data(file2_path)
    
    if df1 is None or df2 is None:
        print("Unable to load data files, program exiting")
        return
    
    # Analyze statistical differences
    stats_df = analyze_statistics(df1, df2, file1_name, file2_name)
    
    # Draw distribution plot and find potential thresholds
    print("Drawing distribution plots and calculating potential thresholds...")
    potential_thresholds = plot_distribution(df1, df2, file1_name, file2_name)
    
    # Output threshold recommendations
    print("\n===== Recommended Thresholds =====")
    for i, (threshold, score) in enumerate(potential_thresholds, 1):
        print(f"Candidate threshold {i}: {threshold:.2f} (Separation score: {score:.2f})")
        
        # Calculate classification effect for each threshold
        correct1 = len(df1[df1['Deflection_Angle'] >= threshold]) / len(df1) * 100
        correct2 = len(df2[df2['Deflection_Angle'] < threshold]) / len(df2) * 100
        
        print(f"  When using threshold {threshold:.2f}:")
        print(f"  - {correct1:.1f}% of samples in {file1_name} will be correctly classified")
        print(f"  - {correct2:.1f}% of samples in {file2_name} will be correctly classified")
        print(f"  - Overall accuracy: {(correct1 + correct2) / 2:.1f}%")
    
    print("\nProgram execution completed. Distribution plot saved as 'Deflection_Angle_distribution.png'")

if __name__ == "__main__":
    main()