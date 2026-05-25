import matplotlib.pyplot as plt
import numpy as np

def create_accuracy_graph():
    """
    Create a simple training and testing accuracy graph
    Training accuracy: 0.95
    Testing accuracy: 0.90
    """
    # Alphabet letters A-Z
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Create consistent accuracies with small variations
    np.random.seed(42)  # For reproducible results
    
    # Training accuracy around 0.95 with small variations
    train_base = 0.95
    train_variations = np.random.uniform(-0.03, 0.03, 26)  # Small random variations
    train_accuracies = np.clip(train_base + train_variations, 0, 1)
    
    # Testing accuracy around 0.90 with small variations
    test_base = 0.90
    test_variations = np.random.uniform(-0.03, 0.03, 26)  # Small random variations
    test_accuracies = np.clip(test_base + test_variations, 0, 1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    x_pos = np.arange(len(alphabet))
    
    # Plot training line (upper, blue)
    plt.plot(x_pos, train_accuracies, 'b-', linewidth=3, 
             label='Training', marker='o', markersize=5, color='#1f77b4')
    
    # Plot testing line (lower, red)
    plt.plot(x_pos, test_accuracies, 'r-', linewidth=3, 
             label='Testing', marker='s', markersize=5, color='#d62728')
    
    # Fill area between curves
    plt.fill_between(x_pos, train_accuracies, test_accuracies, 
                     alpha=0.3, color='lightgray')
    
    # Customize the plot to match your reference
    plt.title('Training and Testing Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Alphabet', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    
    # Set x-axis with all letters
    plt.xticks(x_pos, alphabet, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set y-axis limits
    plt.ylim(0, 1.2)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Add average accuracy text boxes
    avg_train = np.mean(train_accuracies)
    avg_test = np.mean(test_accuracies)
    
    plt.text(0.02, 0.95, f'Avg Training: {avg_train:.3f}', 
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.text(0.02, 0.88, f'Avg Testing: {avg_test:.3f}', 
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('Training_and_Testing_Accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    print("Graph created successfully!")
    print(f"Average Training Accuracy: {avg_train:.3f}")
    print(f"Average Testing Accuracy: {avg_test:.3f}")
    print("Graph saved as 'Training_and_Testing_Accuracy.png'")
    
    plt.show()

if __name__ == "__main__":
    create_accuracy_graph()