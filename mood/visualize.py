import matplotlib.pyplot as plt
import seaborn as sns


def plot_distance_distributions(distances, labels, colors):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
        
    # Visualize all splitting methods
    for idx, dist in enumerate(distances): 
        sns.kdeplot(dist, color=colors[idx], label=labels[idx], ax=ax)
    
    ax.set_xlabel(f"Distance")
    ax.legend()
        
    return ax