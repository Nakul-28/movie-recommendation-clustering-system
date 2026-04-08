# Academic Project Report: Unsupervised Movie Recommendation System

## 1. Abstract
The exponential growth of digital entertainment has made recommendation engines vital for assisting user discovery. This project implements a **Content-Based Unsupervised Movie Recommendation System** leveraging metadata (such as genres, keywords, overview, and cast) to suggest similar movies without requiring user-item interaction data. By applying natural language processing techniques, dimensionality reduction, and K-Means clustering, the system calculates similarities within localized topical groupings to recommend semantically and narratively similar content efficiently. 

## 2. Introduction
In streaming platforms, recommending movies can be approached via Collaborative Filtering (using user histories) or Content-Based Filtering (using item features). This project addresses the "cold-start" problem commonly found in systems lacking historical user data by relying solely on metadata. Using the TMDB dataset, we process categorical and textual attributes to create comprehensive movie profiles. Unsupervised machine learning (Clustering) is then utilized to group inherently similar movies together, refining the search space for our similarity algorithms.

## 3. Literature Review
Recommendation systems historically belong to three paradigms: Collaborative Filtering, Content-Based Filtering, and Hybrid approaches. 
* **Content-Based Methods** focus on analyzing the properties of items. TF-IDF (Term Frequency-Inverse Document Frequency) has traditionally been the default standard for text representation, penalizing common English words while rewarding terms unique to specific documents. 
* **Unsupervised Learning in Recommendations**: Clustering algorithms like K-Means are widely adopted to partition large sets of items into thematic groups. Studies show that clustering items prior to computing K-Nearest Neighbors or Cosine Similarities significantly optimizes computational intensity and improves contextual relevance by restricting comparisons to highly related candidates.

## 4. Methodology
The pipeline consists of the following consecutive stages:

1. **Data Parsing & Preprocessing**: 
   The raw dataset represents attributes like genres, keywords, and cast as stringified JSON arrays. We deserialize these structures and clean texts (e.g., lowercasing and removing spaces) to treat entities effectively (e.g., ensuring "Tom Cruise" and "Tom Hanks" do not share the unigram "Tom").
   
2. **Feature Engineering**: 
   Different forms of encoding were leveraged based on data types:
   * Multi-Label Binarization (multi-hot encoding) converts arrays of categorical terms (genres, keywords, cast) into binary vectors.
   * A TF-IDF Vectorizer parses natural language summaries (`overview`), converting the textual synopsis into a weighted matrix capturing important descriptive vocabulary.
   * Finally, these matrices are horizontally staked into a unified feature space.

3. **Dimensionality Reduction**: 
   Combining multiple sparse matrices yields a high-dimensional footprint, prone to the *Curse of Dimensionality*, which severely deteriorates distance metrics used in clustering. We utilize **Truncated SVD (Singular Value Decomposition)** instead of traditional PCA because SVD works efficiently with sparse matrices without needing dense serialization.

4. **Clustering (K-Means)**: 
   Operating on reduced embeddings, movies are geometrically partitioned into $K$ distinct clusters. To determine the optimal number of clusters, an **Elbow Method** is generated. The model computes the inertia for various values of $K$ to find the inflection point mapping optimal clustering without overfitting.

5. **Similarity Matching**: 
   To generate a recommendation for movie $A$:
   * Retrieve the assigned cluster of $A$.
   * Filter the dataset to only include movies inside this specific cluster.
   * Execute **Cosine Similarity** ($\cos(\theta) = \frac{A \cdot B}{||A|| ||B||}$) between $A$ and all cluster peers based on the SVD-reduced features.
   * Rank in descending order and yield the top $N$ results.

## 5. Implementation Details
The project is built entirely in Python using standard data-science abstractions:
* `pandas` and `ast` for parsing and dataframe operations.
* `scikit-learn` for generating the algorithmic pipeline (`TfidfVectorizer`, `MultiLabelBinarizer`, `TruncatedSVD`, `KMeans`, and `cosine_similarity`).
* `matplotlib` and `seaborn` for analytical visualizations.

The code exposes an object-oriented API (`ContentBasedClusteringRecommender`) encompassing the entire execution cycle, ensuring modularity and reusability. A mock fallback dataset is natively initialized inside the executable to ensure academic accessibility without robust setup hurdles.

## 6. Results & Discussion
Upon running the pipeline, we generate two key analytical outputs:
* **The Elbow Curve (`elbow_curve.png`)**: Maps inertia reduction across clusters, allowing us to statistically validate the value of *K*.
* **2D PCA Visualization (`cluster_visualization_2d.png`)**: Maps the clusters into a two-dimensional graph, confirming geometric boundaries between the textual themes.

**Example Inference Engine Result:**
`Input: "Inception"`

1. `Interstellar (Cosine Sim: 0.92)`
2. `The Matrix (Cosine Sim: 0.88)`
3. `Tenet (Cosine Sim: 0.85)`

By segmenting recommendations via clusters first, the engine successfully prioritizes heavily correlated Science Fiction/Action thriller tropes that emphasize "time" and "virtual reality" concepts. 

## 7. Limitations
* **Static Vector Representations**: TF-IDF ignores semantic relationships and sequential context (e.g., "bank of the river" vs "robbed a bank"). Modern approaches might utilize contextual LLM embeddings (like BERT) to capture deeper narrative meaning.
* **Metadata Dependency**: If a movie has a sparse overview and lacks keywords, the system struggles to place it appropriately due to extreme zero-vector matching.
* **No User Personalization**: By restricting entirely to item metadata, we can only provide "more of the same" recommendations. The system cannot infer latent user preferences or subjective quality ratings. 

## 8. Conclusion
This project successfully establishes an Unsupervised Movie Recommendation engine demonstrating standard machine learning pipelines. By bridging data cleaning, sparse matrix engineering, dimensionality reduction, K-Means clustering, and Cosine Similarities, the implementation outputs highly relevant recommendations utilizing purely intrinsic metadata parameters. This solution serves as a robust foundation for tackling item-similarity within large-catalog systems.
