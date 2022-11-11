import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List


def plot_distance_distributions(
    distances, 
    labels: Optional[List[str]] = None, 
    colors: Optional[List[str]] = None, 
    styles: Optional[List[str]] = None,
    ax: Optional = None
):
    
    n = len(distances)
    show_legend = True
    
    # Set defaults
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if colors is None: 
        cmap = sns.color_palette("rocket", n)
        colors = [cmap[i] for i in range(n)]
    if labels is None: 
        show_legend = False
        labels = [""] * n
    if styles is None: 
        styles = ["-"] * n
    
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
        
    # Visualize all splitting methods
    for idx, dist in enumerate(distances): 
        sns.kdeplot(dist, color=colors[idx], linestyle=styles[idx], ax=ax, label=labels[idx])
    
    ax.set_xlabel(f"Distance")
    
    if show_legend:
        ax.legend()
        
    return ax