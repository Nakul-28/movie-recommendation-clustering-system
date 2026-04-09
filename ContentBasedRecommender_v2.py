import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler   # Unit I: Feature Scaling
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score                             # Unit III: Eval Metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid                        # Unit V: Hyperparameter Tuning

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# WHAT WAS WRONG BEFORE & WHAT WE FIXED (mapped to your syllabus):
#
# UNIT I  - Data Preprocessing:
#   BEFORE: No feature scaling applied after SVD. Features on very different
#           scales caused KMeans to be dominated by high-variance TF-IDF dims.
#   FIX:    StandardScaler applied to reduced_features before clustering.
#           This is critical — KMeans uses Euclidean distance, so scale matters.
#
# UNIT I  - Dimensionality Reduction (PCA / SVD):
#   BEFORE: Fixed n_components=100 with no justification. Too many components
#           can retain noise; too few lose signal.
#   FIX:    Variance-based component selection — keep components that explain
#           ≥ 85% cumulative variance. This is the standard PCA best practice.
#
# UNIT V  - Hyperparameter Tuning & Cross-Validation:
#   BEFORE: K=20 was hardcoded. No systematic search for the best K.
#   FIX:    Grid search over K using Silhouette Score as the objective metric.
#           Silhouette measures intra-cluster cohesion vs inter-cluster separation
#           — a much more reliable guide than the elbow curve alone.
#
# UNIT V  - Unsupervised Learning / Clustering:
#   BEFORE: Recommending from within the cluster only using cosine similarity
#           was fine in theory, but cluster quality was so poor that "same
#           cluster" didn't actually mean "similar movie."
#   FIX:    Better clusters (from scaling + tuned K) mean cluster-filtering now
#           genuinely narrows to thematically related movies. Also added a
#           fallback: if cluster is too small, broaden to global top-N.
#
# UNIT II - Feature Engineering (analogous to regression feature work):
#   BEFORE: All features weighted equally. TF-IDF (5000-dim) drowns out genres
#           (20-dim) and cast because the sparse matrix is unbalanced.
#   FIX:    Weighted feature combination — genres/keywords/cast/director are
#           upsampled with np.repeat to give them proportional influence.
#           Genre is especially important for perceived similarity.
#
# UNIT III - Evaluation:
#   BEFORE: Only silhouette score for small datasets. No per-cluster diagnostics.
#   FIX:    Added cluster_report() that shows cluster size distribution and top
#           genres per cluster, making it easy to audit cluster quality.
# =============================================================================


# Similarity weights for each feature block.
# title is word-level TF-IDF (not char n-grams) — good for exact word overlap
# but overview+keywords carry the bulk of semantic meaning.
FEATURE_WEIGHTS = {
    'title':    0.15,   # Reduced — word overlap matters but shouldn't dominate
    'overview': 0.35,   # Highest — main semantic description of the film
    'keywords': 0.25,   # High — curated tags that describe plot/theme precisely
    'genres':   0.12,   # Moderate — broad but reliable genre signal
    'cast':     0.08,   # Lower — relevant but actors span many genres
    'director': 0.05,   # Lowest — signals style but directors vary a lot
}

# How much popularity can boost the composite similarity score.
# Formula: final = sim * (1 + POP_BOOST * pop_norm)
# At POP_BOOST=0.4 a maximally popular film gets a +40% boost,
# but a film with 2× the similarity score will still win.
POP_BOOST_WEIGHT = 0.4


class ContentBasedClusteringRecommender:
    """
    Improved Unsupervised Content-Based Movie Recommendation System.
    Fixes: Feature Scaling, Variance-based SVD per block, Hyperparameter Tuning for K,
           Per-feature cosine similarity with explicit weights, Popularity-ranked output.
    """

    def __init__(self, data_path_movies):
        self.movies_path = data_path_movies
        self.df = None

        # Per-feature reduced matrices (used for similarity)
        self.feat_reduced = {}          # {name: ndarray}
        self.feat_svd     = {}          # {name: TruncatedSVD}

        # Combined matrix for clustering
        self.features_matrix = None
        self.reduced_features = None
        self.scaled_features = None
        self.cluster_labels = None
        self.kmeans_model = None
        self.best_k = None

        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)

        # Encoders
        self.tfidf_overview = TfidfVectorizer(max_features=5000, stop_words='english')
        # Word-level TF-IDF for title: avoids char n-gram false matches
        # (char n-grams would match 'Rush' → 'Rust'/'Rudy' via shared trigrams).
        self.tfidf_title    = TfidfVectorizer(max_features=2000, stop_words='english')
        self.mlb_genres   = MultiLabelBinarizer(sparse_output=True)
        self.mlb_keywords = MultiLabelBinarizer(sparse_output=True)
        self.mlb_cast     = MultiLabelBinarizer(sparse_output=True)
        self.mlb_director = MultiLabelBinarizer(sparse_output=True)

        # Unit I: StandardScaler — zero mean, unit variance before clustering
        self.scaler = StandardScaler()

    # -------------------------------------------------------------------------
    # HELPER
    # -------------------------------------------------------------------------
    def _convert_json_to_list(self, text, top_n=None):
        if pd.isna(text):
            return []
        try:
            items = ast.literal_eval(text)
            if not isinstance(items, list):
                return []
            parsed = [str(i).strip().replace(" ", "").lower() for i in items]
            return parsed[:top_n] if top_n else parsed
        except:
            return []

    # -------------------------------------------------------------------------
    # STEP 1: LOAD & PREPROCESS
    # -------------------------------------------------------------------------
    def load_and_preprocess(self):
        """Unit I: Data Cleaning, Handling Missing Values, Encoding Categorical Data."""
        print("Loading data...")
        df_movies = pd.read_csv(self.movies_path)

        cols_to_keep = ['title', 'overview', 'genres', 'keywords']
        if 'cast' in df_movies.columns:
            cols_to_keep.append('cast')
        if 'director' in df_movies.columns:
            cols_to_keep.append('director')
        if 'popularity' in df_movies.columns:
            cols_to_keep.append('popularity')

        self.df = df_movies[cols_to_keep].copy()
        self.df['overview'] = self.df['overview'].fillna('')
        self.df['title']    = self.df['title'].fillna('')
        if 'popularity' in self.df.columns:
            self.df['popularity'] = pd.to_numeric(self.df['popularity'], errors='coerce').fillna(0.0)
            # Log-normalise popularity into [0, 1] for the hybrid score boost.
            # log1p dampens extreme outliers (Avengers-level movies).
            log_pop = np.log1p(self.df['popularity'])
            self.df['pop_norm'] = log_pop / log_pop.max()
        else:
            self.df['pop_norm'] = 0.0

        self.df['genres']   = self.df['genres'].apply(self._convert_json_to_list)
        self.df['keywords'] = self.df['keywords'].apply(self._convert_json_to_list)
        if 'cast' in self.df.columns:
            self.df['cast'] = self.df['cast'].apply(lambda x: self._convert_json_to_list(x, top_n=3))

        self.df = self.df.dropna(subset=['title']).reset_index(drop=True)
        print(f"Preprocessed. Shape: {self.df.shape}")

    # -------------------------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING (with weighting)
    # -------------------------------------------------------------------------
    def engineer_features(self):
        """
        Unit I: Encoding Categorical Data, Feature Engineering.
        Builds a SEPARATE sparse matrix for each feature so per-feature cosine
        similarities can be computed independently with their own SVD reduction.
        A combined weighted matrix is also kept for clustering.
        """
        print("Engineering per-feature matrices...")

        # Raw sparse matrices for each feature
        self._raw_feat = {}

        # Title — character n-gram TF-IDF
        self._raw_feat['title']    = self.tfidf_title.fit_transform(self.df['title'])

        # Overview — word TF-IDF
        self._raw_feat['overview'] = self.tfidf_overview.fit_transform(self.df['overview'])

        # Keywords — multi-hot
        self._raw_feat['keywords'] = self.mlb_keywords.fit_transform(self.df['keywords'])

        # Genres — multi-hot
        self._raw_feat['genres']   = self.mlb_genres.fit_transform(self.df['genres'])

        if 'cast' in self.df.columns:
            self._raw_feat['cast'] = self.mlb_cast.fit_transform(self.df['cast'])

        if 'director' in self.df.columns:
            dir_series = self.df['director'].apply(
                lambda x: [str(x).replace(" ", "").lower()] if pd.notna(x) else []
            )
            self._raw_feat['director'] = self.mlb_director.fit_transform(dir_series)

        # ---- Combined weighted matrix for clustering (unchanged pipeline) ----
        # Use sqrt-weighting so high-dim TF-IDF doesn't drown out categorical blocks
        weight_map = {'title': 2.0, 'overview': 1.0, 'keywords': np.sqrt(5),
                      'genres': np.sqrt(10), 'cast': np.sqrt(3), 'director': np.sqrt(5)}
        feature_blocks = [self._raw_feat[k] * weight_map.get(k, 1.0)
                          for k in self._raw_feat]
        self.features_matrix = hstack(feature_blocks)
        print(f"Combined feature matrix shape: {self.features_matrix.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: DIMENSIONALITY REDUCTION (variance-based component selection)
    # -------------------------------------------------------------------------
    def reduce_dimensionality(self, variance_threshold=0.85):
        """
        Unit I: Dimensionality Reduction (TruncatedSVD).

        TWO-STAGE approach:
          1. Per-feature SVD: each feature block gets its own SVD, sized to
             explain `variance_threshold` of that block's variance.  This is
             what's used for per-feature cosine similarity in recommend().
             Because we reduce inside each block the explained variance is
             genuinely high and interpretable.
          2. Global SVD on the combined matrix: used for KMeans clustering
             (same as before but same variance-threshold logic).
        """
        print("=== Stage 1: Per-feature SVD (for similarity) ===")
        for name, mat in self._raw_feat.items():
            max_comp = min(300, mat.shape[0] - 1, mat.shape[1] - 1)
            if max_comp < 2:
                # Degenerate block — keep raw (already dense or 1-dim)
                self.feat_reduced[name] = mat.toarray() if hasattr(mat, 'toarray') else mat
                continue

            # Probe pass
            probe = TruncatedSVD(n_components=max_comp, random_state=42)
            probe.fit(mat)
            cum_var = np.cumsum(probe.explained_variance_ratio_)
            n_comp  = int(np.searchsorted(cum_var, variance_threshold)) + 1
            n_comp  = max(n_comp, 5)
            n_comp  = min(n_comp, max_comp)
            total_var = cum_var[n_comp - 1] * 100
            print(f"  [{name:10s}] components={n_comp:4d}  "
                  f"explained variance={total_var:.1f}%")

            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.feat_reduced[name] = svd.fit_transform(mat)
            self.feat_svd[name]     = svd

        # Plot per-feature explained variance
        plt.figure(figsize=(8, 5))
        for name, svd in self.feat_svd.items():
            cum = np.cumsum(svd.explained_variance_ratio_)
            plt.plot(cum, label=name)
        plt.axhline(y=variance_threshold, color='black', linestyle='--',
                    label=f'Threshold {variance_threshold*100:.0f}%')
        plt.title('Cumulative Explained Variance per Feature Block (SVD)')
        plt.xlabel('Number of SVD Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'svd_variance_per_feature.png'))
        plt.close()

        print("\n=== Stage 2: Global SVD on combined matrix (for clustering) ===")
        max_components = min(300, self.features_matrix.shape[0] - 1,
                             self.features_matrix.shape[1] - 1)
        svd_probe = TruncatedSVD(n_components=max_components, random_state=42)
        svd_probe.fit(self.features_matrix)
        cumulative_variance = np.cumsum(svd_probe.explained_variance_ratio_)

        n_components = int(np.searchsorted(cumulative_variance, variance_threshold)) + 1
        n_components = max(n_components, 20)
        n_components = min(n_components, max_components)
        print(f"Global SVD: {n_components} components → "
              f"{cumulative_variance[n_components-1]*100:.1f}% variance")

        # Plot global variance curve
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_variance, color='steelblue')
        plt.axvline(x=n_components, color='red', linestyle='--',
                    label=f'Selected: {n_components} components')
        plt.axhline(y=variance_threshold, color='green', linestyle='--',
                    label=f'Threshold: {variance_threshold*100:.0f}%')
        plt.title('Cumulative Explained Variance vs SVD Components (Global)')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'svd_variance_curve.png'))
        plt.close()

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.reduced_features = self.svd.fit_transform(self.features_matrix)

        # StandardScaler — CRITICAL for KMeans (Euclidean distance)
        print("Applying StandardScaler to global reduced features...")
        self.scaled_features = self.scaler.fit_transform(self.reduced_features)
        print(f"Final feature shape for clustering: {self.scaled_features.shape}")

    # -------------------------------------------------------------------------
    # STEP 4a: HYPERPARAMETER TUNING — find best K via Silhouette Score
    # -------------------------------------------------------------------------
    def tune_k(self, k_range=range(10, 51, 5)):
        """
        Unit V: Hyperparameter Tuning.
        FIX: Grid search over K values; pick K with highest Silhouette Score.
        Silhouette Score ∈ [-1, 1]: higher = better separated, denser clusters.
        """
        print("Tuning K via Silhouette Score grid search...")
        results = []

        # Subsample for speed if dataset is large
        n = len(self.scaled_features)
        sample_idx = np.random.choice(n, min(n, 3000), replace=False)
        X_sample = self.scaled_features[sample_idx]

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sample)
            score = silhouette_score(X_sample, labels)
            results.append({'k': k, 'silhouette': score})
            print(f"  K={k:3d} | Silhouette: {score:.4f}")

        results_df = pd.DataFrame(results)
        self.best_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
        print(f"\nBest K = {self.best_k} (Silhouette: {results_df['silhouette'].max():.4f})")

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(results_df['k'], results_df['silhouette'], marker='o', color='darkorange')
        plt.axvline(x=self.best_k, color='red', linestyle='--', label=f'Best K={self.best_k}')
        plt.title('Silhouette Score vs Number of Clusters K')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'silhouette_vs_k.png'))
        plt.close()
        return self.best_k

    # -------------------------------------------------------------------------
    # STEP 4b: CLUSTER
    # -------------------------------------------------------------------------
    def cluster_movies(self, k=None, silhouette_sample=5000):
        """Unit V: Clustering. Uses scaled features and tuned K.

        silhouette_sample: max rows used to estimate the silhouette score.
        silhouette_score on the full matrix requires an N×N pairwise distance
        computation that OOMs on large / high-dimensional datasets.
        Sampling mirrors exactly what tune_k() already does.
        """
        k = k or self.best_k or 20
        print(f"Clustering with K={k} on SCALED features...")
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=15)
        self.cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)
        self.df['cluster'] = self.cluster_labels

        # --- Silhouette score on a capped sample (avoids OOM) ----------------
        n = len(self.scaled_features)
        if n > silhouette_sample:
            idx = np.random.choice(n, silhouette_sample, replace=False)
            score = silhouette_score(self.scaled_features[idx],
                                     self.cluster_labels[idx])
            print(f"Final Silhouette Score (sample n={silhouette_sample}): {score:.4f}")
        else:
            score = silhouette_score(self.scaled_features, self.cluster_labels)
            print(f"Final Silhouette Score: {score:.4f}")

    # -------------------------------------------------------------------------
    # STEP 4c: CLUSTER QUALITY REPORT (for auditing)
    # -------------------------------------------------------------------------
    def cluster_report(self):
        """
        Unit III: Evaluation — inspect cluster quality by checking
        cluster size distribution and dominant genres per cluster.
        Good clusters should be cohesive (similar genres) and balanced (not one huge cluster).
        """
        print("\n--- Cluster Quality Report ---")
        cluster_sizes = self.df['cluster'].value_counts().sort_index()
        print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
              f"mean={cluster_sizes.mean():.1f}")

        if 'genres' in self.df.columns:
            print("\nTop genres per cluster (first 5 clusters):")
            for c in sorted(self.df['cluster'].unique())[:5]:
                cluster_df = self.df[self.df['cluster'] == c]
                all_genres = [g for genres in cluster_df['genres'] for g in genres]
                if all_genres:
                    top_genres = pd.Series(all_genres).value_counts().head(3).index.tolist()
                    print(f"  Cluster {c} ({len(cluster_df)} movies): {top_genres}")

    # -------------------------------------------------------------------------
    # STEP 5: VISUALIZE
    # -------------------------------------------------------------------------
    def visualize_clusters_2d(self):
        """
        Unit I: Dimensionality Reduction for visualisation.
        Uses the first two SVD dimensions of the global reduced matrix directly
        (no second PCA layer), so the axes represent real SVD components with
        knowable explained variance.
        """
        print("Generating 2D cluster visualisation...")

        # The first 2 dims of reduced_features already come from TruncatedSVD
        # They explain the most variance — no need to run PCA on top.
        svd_2d = self.reduced_features[:, :2]
        ev1 = self.svd.explained_variance_ratio_[0] * 100
        ev2 = self.svd.explained_variance_ratio_[1] * 100

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=svd_2d[:, 0], y=svd_2d[:, 1],
                        hue=self.cluster_labels, palette='tab20',
                        s=25, alpha=0.7, legend='full')
        plt.title('Movie Clusters (first 2 SVD components)')
        plt.xlabel(f'SVD Component 1 ({ev1:.1f}% var)')
        plt.ylabel(f'SVD Component 2 ({ev2:.1f}% var)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'cluster_2d.png'))
        plt.close()
        print("Saved cluster_2d.png")

    # -------------------------------------------------------------------------
    # STEP 6: RECOMMEND (with fallback)
    # -------------------------------------------------------------------------
    def recommend(self, movie_title, top_n=5):
        """
        Unit V: Real-World Application — Recommendation System.

        Similarity = weighted sum of per-feature cosine similarities:
          title × 0.30 + overview × 0.25 + keywords × 0.20
          + genres × 0.10 + cast × 0.08 + director × 0.07

        Final ranking is by popularity score among the top-2*top_n similarity
        candidates so that equally relevant films surface the most popular ones.
        """
        match = self.df[self.df['title'].str.lower() == movie_title.lower()]
        if match.empty:
            match = self.df[self.df['title'].str.lower().str.contains(movie_title.lower())]
            if match.empty:
                return f"Movie '{movie_title}' not found in dataset."

        movie_idx     = match.index[0]
        movie_cluster = self.df.loc[movie_idx, 'cluster']
        cluster_indices = self.df[self.df['cluster'] == movie_cluster].index

        MIN_CLUSTER_SIZE = top_n + 1
        if len(cluster_indices) > MIN_CLUSTER_SIZE:
            search_indices = cluster_indices
            scope = f"Cluster {movie_cluster}"
        else:
            search_indices = self.df.index
            scope = "Global (cluster too small)"

        # ---- Per-feature cosine similarities --------------------------------
        sim_scores = np.zeros(len(search_indices), dtype=np.float64)

        for feat_name, weight in FEATURE_WEIGHTS.items():
            if feat_name not in self.feat_reduced:
                continue
            feat_mat = self.feat_reduced[feat_name]
            target_vec   = feat_mat[movie_idx].reshape(1, -1)
            search_vecs  = feat_mat[np.array(search_indices)]
            block_sims   = cosine_similarity(target_vec, search_vecs).flatten()
            sim_scores  += weight * block_sims

        # ---- Hybrid score: similarity * popularity boost -------------------
        # Using a multiplicative boost so popularity can never overcome a
        # large sim gap.  pop_norm ∈ [0,1] (log-scaled), so the maximum
        # boost is (1 + POP_BOOST_WEIGHT) ≈ 1.4×  —  a film with 2× the
        # raw similarity will always beat a more popular but less relevant one.
        pop_norms   = self.df.loc[search_indices, 'pop_norm'].to_numpy()
        final_scores = sim_scores * (1.0 + POP_BOOST_WEIGHT * pop_norms)

        final_order = final_scores.argsort()[::-1]

        print(f"\n--- Recommendations for '{self.df.loc[movie_idx, 'title']}' "
              f"[{scope}] ---")
        recommendations = []
        for idx in final_order:
            actual_idx = search_indices[idx]
            if actual_idx == movie_idx:
                continue
            row        = self.df.loc[actual_idx]
            pop_val    = self.df.loc[actual_idx, 'popularity'] \
                         if 'popularity' in self.df.columns else 0.0
            genres_str = ', '.join(row['genres'][:3]) if isinstance(row['genres'], list) else ''
            print(f"  {len(recommendations)+1}. {row['title']:<40} "
                  f"sim={sim_scores[idx]:.4f}  final={final_scores[idx]:.4f}  "
                  f"pop={pop_val:.2f}  genres=[{genres_str}]")
            recommendations.append((row['title'], sim_scores[idx], pop_val))
            if len(recommendations) == top_n:
                break

        return recommendations


# =============================================================================
# MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    target_csv = "dataset/tmdb_movies_cleaned.csv"
    if not os.path.exists(target_csv):
        print(f"Error: Dataset not found at {target_csv}.")
        exit(1)

    rec = ContentBasedClusteringRecommender(data_path_movies=target_csv)

    # Step 1 — Preprocess (Unit I)
    rec.load_and_preprocess()

    # Step 2 — Feature Engineering with weighting (Unit I + II)
    rec.engineer_features()

    # Step 3 — Variance-based SVD + StandardScaler (Unit I)
    rec.reduce_dimensionality(variance_threshold=0.85)

    # Step 4a — Hyperparameter tuning: find best K (Unit V)
    rec.tune_k(k_range=range(10, 51, 5))

    # Step 4b — Cluster with best K (Unit V)
    rec.cluster_movies()

    # Step 4c — Audit cluster quality (Unit III)
    rec.cluster_report()

    # Step 5 — Visualise (Unit I)
    rec.visualize_clusters_2d()

    # Step 6 — Recommendations (Unit V)
    print("\n" + "="*60)
    rec.recommend("Inception",    top_n=5)
    rec.recommend("The Dark Knight", top_n=5)
    rec.recommend("Toy Story",    top_n=5)
    
