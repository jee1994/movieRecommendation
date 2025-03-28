import torch
import numpy as np

def visualize_contrastive_effect(model, features, user_data, output_file='contrastive_effect.png'):
    """Visualize the effect of contrastive learning on user embeddings"""
    # Generate embeddings with and without contrastive learning
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        # Original embeddings
        embeddings_original = model(features_tensor)
        embeddings_original = nn.functional.normalize(embeddings_original, p=2, dim=1).numpy()
        
        # Apply t-SNE to reduce dimensionality
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_original)
        
        # Create visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        
        # Plot by age and gender
        age_groups = [(0, 25), (25, 35), (35, 45), (45, 100)]
        genders = ['M', 'F']
        
        # Create a 2x2 grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot by gender
        for gender, color in zip(genders, ['blue', 'red']):
            mask = user_data['Gender'] == gender
            axes[0, 0].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=f'Gender: {gender}',
                alpha=0.7
            )
        axes[0, 0].set_title('User Embeddings by Gender')
        axes[0, 0].legend()
        
        # Plot by age
        colors = ['green', 'orange', 'purple', 'brown']
        for i, ((min_age, max_age), color) in enumerate(zip(age_groups, colors)):
            mask = (user_data['Age'] >= min_age) & (user_data['Age'] < max_age)
            axes[0, 1].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=f'Age: {min_age}-{max_age}',
                alpha=0.7
            )
        axes[0, 1].set_title('User Embeddings by Age Group')
        axes[0, 1].legend()
        
        # Plot by gender and age combined
        for i, (min_age, max_age) in enumerate(age_groups):
            for j, gender in enumerate(genders):
                mask = (user_data['Gender'] == gender) & (user_data['Age'] >= min_age) & (user_data['Age'] < max_age)
                marker = 'o' if gender == 'M' else '^'
                axes[1, 0].scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    marker=marker,
                    label=f'{gender}, {min_age}-{max_age}',
                    alpha=0.7
                )
        axes[1, 0].set_title('User Embeddings by Gender and Age')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot embedding distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings_original))
        
        # Plot distance histogram
        axes[1, 1].hist(distances[np.triu_indices_from(distances, k=1)], bins=50)
        axes[1, 1].set_title('Distribution of User Embedding Distances')
        axes[1, 1].set_xlabel('Distance')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_file)
        
        return embeddings_2d, distances 