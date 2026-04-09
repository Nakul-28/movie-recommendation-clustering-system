import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

class ContentBasedClusteringRecommender:
    """
    Unsupervised Content-Based Movie Recommendation System.
    Uses Dimensionality Reduction, Clustering, and Cosine Similarity on Movie Metadata.
    """
    def __init__(self, data_path_movies, data_path_credits=None):
        self.movies_path = data_path_movies
        self.credits_path = data_path_credits
        self.df = None
        self.features_matrix = None
        self.reduced_features = None
        self.cluster_labels = None
        self.kmeans_model = None
        
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Encoders and scalers
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_title = TfidfVectorizer(max_features=5000, stop_words='english')
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_keywords = MultiLabelBinarizer()
        self.mlb_cast = MultiLabelBinarizer()
        self.mlb_director = MultiLabelBinarizer()
        self.svd = TruncatedSVD(n_components=100, random_state=42)

    def _convert_json_to_list(self, text, top_n=None):
        """Helper to convert stringified lists of strings locally."""
        if pd.isna(text):
            return []
        try:
            items = ast.literal_eval(text)
            if not isinstance(items, list):
                return []
            parsed_list = [str(item).strip().replace(" ", "").lower() for item in items]
            if top_n:
                parsed_list = parsed_list[:top_n]
            return parsed_list
        except:
            return []

    def load_and_preprocess(self):
        """
        Step 1: Data Preprocessing
        Loads dataset, handles missing values, and processes text & JSON columns.
        """
        print("Loading data...")
        df_movies = pd.read_csv(self.movies_path)
        
        # Features are pre-merged in our cleaned dataset.

        # Select relevant columns 
        cols_to_keep = ['title', 'overview', 'genres', 'keywords']
        if 'cast' in df_movies.columns:
            cols_to_keep.append('cast')
        if 'director' in df_movies.columns:
            cols_to_keep.append('director')
        if 'popularity' in df_movies.columns:
            cols_to_keep.append('popularity')
            
        self.df = df_movies[cols_to_keep].copy()
        
        # Handle missing values
        self.df['overview'] = self.df['overview'].fillna('')
        
        print("Preprocessing JSON columns...")
        # Parse JSON to lists, clean text (lowercase, remove spaces for entities)
        self.df['genres'] = self.df['genres'].apply(self._convert_json_to_list)
        self.df['keywords'] = self.df['keywords'].apply(self._convert_json_to_list)
        
        if 'cast' in self.df.columns:
            # Keep top 3 cast members to avoid noise
            self.df['cast'] = self.df['cast'].apply(lambda x: self._convert_json_to_list(x, top_n=3))

        # Drop rows where title is missing (if any)
        self.df = self.df.dropna(subset=['title']).reset_index(drop=True)
        print(f"Data preprocessed successfully. Shape: {self.df.shape}")

    def engineer_features(self):
        """
        Step 2: Feature Engineering
        Converts text and categorical multi-labels into a unified numerical sparse matrix.
        """
        print("Engineering features...")
        
        self.feature_matrices = {}
        
        # 0. TF-IDF on Title
        self.feature_matrices['title'] = self.tfidf_title.fit_transform(self.df['title'])
        
        # 1. TF-IDF on Overview
        self.feature_matrices['overview'] = self.tfidf.fit_transform(self.df['overview'])
        
        # 2. Multi-hot Encoding on Genres
        self.feature_matrices['genres'] = self.mlb_genres.fit_transform(self.df['genres'])
        
        # 3. Multi-hot Encoding on Keywords
        self.feature_matrices['keywords'] = self.mlb_keywords.fit_transform(self.df['keywords'])
        
        # 4. Multi-hot Encoding on Cast (Optional but included)
        if 'cast' in self.df.columns:
            self.feature_matrices['cast'] = self.mlb_cast.fit_transform(self.df['cast'])
            
        # 5. Multi-hot Encoding on Director
        if 'director' in self.df.columns:
            director_series = self.df['director'].apply(lambda x: [str(x).replace(" ", "").lower()] if pd.notna(x) else [])
            self.feature_matrices['director'] = self.mlb_director.fit_transform(director_series)
            
        # Combine all features into a single sparse matrix
        self.features_matrix = hstack(list(self.feature_matrices.values()))
        print(f"Feature matrix created. Shape: {self.features_matrix.shape}")

    def reduce_dimensionality(self):
        """
        Step 3: Dimensionality Reduction
        TruncatedSVD is used as opposed to PCA because our matrix is sparse.
        This handles the "Curse of Dimensionality" and improves clustering performance.
        """
        print("Applying Truncated SVD for dimensionality reduction...")
        # Reduce to 100 components to capture maximum variance while reducing noise
        n_comps = min(100, self.features_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comps, random_state=42)
        self.reduced_features = self.svd.fit_transform(self.features_matrix)
        
        explained_variance = sum(self.svd.explained_variance_ratio_) * 100
        print(f"Dimensionality reduced to {n_comps} dimensions. Explained Variance: {explained_variance:.2f}%")

    def find_optimal_clusters_elbow(self, max_k=30, step=5):
        """
        Step 4a: Evaluate Elbow Method to help find optimal K.
        Saves visual plot to local directory.
        """
        print("Computing Elbow Curve...")
        distortions = []
        K_range = range(5, max_k + 1, step)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.reduced_features)
            distortions.append(kmeans.inertia_)
            
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, distortions, marker='o', linestyle='-', color='b')
        plt.title('Elbow Method For Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia (Sum of squared distances)')
        plt.grid(True)
        plot_path = os.path.join(self.plots_dir, 'elbow_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to '{plot_path}'.")

    def cluster_movies(self, k=20):
        """
        Step 4b: Perform K-Means Clustering.
        """
        print(f"Clustering with K={k}...")
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(self.reduced_features)
        self.df['cluster'] = self.cluster_labels
        
        # Optional: Compute Silhouette Score (computationally heavy for large datasets)
        if len(self.reduced_features) <= 5000:
            score = silhouette_score(self.reduced_features, self.cluster_labels)
            print(f"Silhouette Score: {score:.4f}")

    def visualize_clusters_2d(self):
        """
        Step 6a: 2D Cluster Visualization via PCA.
        """
        print("Generating 2D cluster visualization...")
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(self.reduced_features)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pca_2d[:,0], y=pca_2d[:,1], hue=self.cluster_labels, 
                        palette='tab20', s=30, alpha=0.7, legend='full')
        plt.title('Movie Clusters mapped to 2D using PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'cluster_visualization_2d.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to '{plot_path}'.")

    def recommend(self, movie_title, top_n=5):
        """
        Step 5: Recommendation System Logic
        Recommends similar movies operating exclusively within the target movie's cluster.
        """
        # Find exactly the requested movie
        match = self.df[self.df['title'].str.lower() == movie_title.lower()]
        
        if match.empty:
            return f"Movie '{movie_title}' not found in the dataset."
            
        movie_idx = match.index[0]
        movie_cluster = self.df.loc[movie_idx, 'cluster']
        
        # 1. Filter movies to only those in the same cluster
        cluster_indices = self.df[self.df['cluster'] == movie_cluster].index
        
        if len(cluster_indices) <= 1:
            return "Not enough movies in the cluster to make a recommendation."

        # 2. Compute Cosine Similarity as a weighted sum of individual feature similarities
        similarities = np.zeros(len(cluster_indices))
        
        # Adjust coefficients where title, keywords, overview: highest priority 
        # genre, cast, director: slightly lower
        coefficients = {
            'title': 2.0,
            'keywords': 2.0,
            'overview': 2.0,
            'genres': 1.0,
            'cast': 1.0,
            'director': 1.0
        }
        
        for feat_name, feat_matrix in self.feature_matrices.items():
            coef = coefficients.get(feat_name, 1.0)
            target_feat = feat_matrix[movie_idx]
            cluster_feat = feat_matrix[cluster_indices]
            sim = cosine_similarity(target_feat, cluster_feat).flatten()
            similarities += coef * sim
            
        print(f"\n--- Recommendations for '{match['title'].values[0]}' (Cluster: {movie_cluster}) ---")
        
        # 3. Sort by similarity to find top candidates, then rank by popularity
        candidates = []
        for i, actual_idx in enumerate(cluster_indices):
            if actual_idx != movie_idx: # Skip the movie itself
                candidates.append({
                    'title': self.df.loc[actual_idx, 'title'],
                    'similarity': similarities[i],
                    'popularity': self.df.loc[actual_idx, 'popularity'] if 'popularity' in self.df.columns else 0.0
                })
        
        # Filter top similar candidates first, then sort by popularity
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        top_candidates = candidates[:max(top_n * 4, 20)]
        top_candidates.sort(key=lambda x: x['popularity'], reverse=True)
        
        recommendations = []
        for i, cand in enumerate(top_candidates[:top_n]):
            sim_movie_title = cand['title']
            similarity_score = cand['similarity']
            pop_score = cand['popularity']
            recommendations.append((sim_movie_title, similarity_score, pop_score))
            print(f"{i+1}. {sim_movie_title} (Weighted Sim: {similarity_score:.4f}, Popularity: {pop_score:.2f})")
                    
        return recommendations


# Example usage block to execute the pipeline
if __name__ == "__main__":
    target_csv = "dataset/tmdb_movies_cleaned.csv"
    if not os.path.exists(target_csv):
        print(f"Error: Dataset not found at {target_csv}. Please run Data_Cleaning.py first.")
        exit(1)

    # Step 1-3: Initialization and Preprocessing
    recommender = ContentBasedClusteringRecommender(data_path_movies=target_csv)
    recommender.load_and_preprocess()
    recommender.engineer_features()
    recommender.reduce_dimensionality()
    
    # Step 4: Clustering & Evaluation
    eval_k = 20
    elbow_max = 30
    elbow_step = 5
    
    recommender.find_optimal_clusters_elbow(max_k=elbow_max, step=elbow_step)
    recommender.cluster_movies(k=eval_k)
    recommender.visualize_clusters_2d()
    
    # Step 5: Inference
    print("\n------------------------------")
    recommender.recommend("Inception")
    print("------------------------------")
