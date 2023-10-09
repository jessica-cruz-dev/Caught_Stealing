import matplotlib.pyplot as plt

def elbow_curve(x, y):
    """Plot simple elbow curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()