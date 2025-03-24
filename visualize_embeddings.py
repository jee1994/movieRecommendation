def visualize_user_embeddings(embeddings, user_data, output_file='user_embeddings.png'):
    """Visualize user embeddings colored by demographics"""
    # Reduce to 2D for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    
    # Color by gender
    colors = {'M': 'blue', 'F': 'red'}
    for gender in ['M', 'F']:
        mask = user_data['Gender'] == gender
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[gender],
            label=f'Gender: {gender}',
            alpha=0.7
        )
    
    plt.title('User Embeddings Visualization')
    plt.legend()
    plt.savefig(output_file)
    
    # Also create age-based visualization
    plt.figure(figsize=(12, 10))
    age_groups = [(0, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 100)]
    for i, (min_age, max_age) in enumerate(age_groups):
        mask = (user_data['Age'] >= min_age) & (user_data['Age'] < max_age)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f'Age: {min_age}-{max_age}',
            alpha=0.7
        )
    
    plt.title('User Embeddings by Age Group')
    plt.legend()
    plt.savefig('user_embeddings_by_age.png') 