# Music Maven: Artist Similarity Classification
For this subsection of 'Music Maven', we propose the creation of an artist classification and similarity search system. Given a song's audio features, the system should be able to identify the most likely artist, and given an artist, find other artists with similar musical profiles. To achieve this, we will build artist-level feature representations from MIR data and apply similarity and classification methods to enable queries such as:

* "Who makes music that sounds like this song?"
* "Find me artists similar to x."
* "Which artists are most similar to this one in terms of energy and tempo?"

The system will operate as a module within the larger Music Maven architecture, allowing the central chatbot to route artist-related queries to our service.

The high-level design consists of three components. First, an **artist profiling** step that aggregates per-song MIR features from the `Music4All` dataset into representative per-artist feature vectors. Second, an **artist similarity** component that computes distances between artist profiles to return a ranked list of similar artists given a query. Third, an **artist classifier** that, given a song's feature vector, identifies the most probable artist. These two capabilities — similarity and classification — naturally complement each other and share the same underlying feature representations.

For artist similarity, we will use a weighted distance measure to compare artist profiles across a selected subset of features, enabling queries such as "find artists similar in energy and tempo but exclude pop." For artist classification, we will use a k-Nearest Neighbors (k-NN) approach, classifying a query song by finding its closest neighbours in feature space and returning the most common artist label.

To tie the system into the Music Maven conversational interface, we will also explore a query language (JSON-based) that maps structured user constraints to similarity parameters. Future work includes translating natural language input — such as "artists like Pink Floyd but more electronic" — via an LLM into a structured query.

# Timeline & Objectives

### Milestone 1 – Import Dataset, Initialize Repository (Goal: 2026-02-17)

Obtain a local copy of the `Music4All` or other datasets. Set up the repository structure and ensure all team members can push and pull cleanly. Familiarize ourselves with the dataset schema — specifically the per-song MIR features (tempo, energy, valence, key, mode, danceability) and associated metadata (artist ID, album, release year, genres, tags).

### Milestone 2 – Artist Profiling (Goal: 2026-02-24)

Construct artist-level feature vectors by aggregating per-song MIR features across all songs attributed to each artist. The primary aggregation method will be the mean across songs per artist, with optional variance or median alternatives to capture spread. Store these artist profiles in a structured `Pandas` DataFrame for downstream querying and training.

**MIR Features in M4A used for artist profiling:**
- Tempo
- Energy
- Valence
- Danceability
- Key
- Mode

**Metadata Features:**
- Artist ID / Name
- Genres
- Tags
- Release Date range (min/max year per artist)

### Milestone 3 – Artist Similarity Search (Goal: 2026-03-03)

Implement a similarity search over artist profiles. Given a query artist, compute a ranked list of the most similar artists using a weighted distance measure. We will explore both **Weighted Euclidean Distance** and **Cosine Similarity** to compare their effectiveness across different feature subsets.

The formula for Weighted Euclidean Distance between two artist profiles $\mathbf{x}$ and $\mathbf{y}$ is:

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{N} w_i (x_i - y_i)^2}
$$

where $w_i$ is the user-specified importance weight for feature $i$.

The formula for Cosine Similarity is:

$$
\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$

This step will also involve normalizing artist profiles so that numeric features are on a comparable scale before comparison.

### Milestone 4 – Artist Classification, Query Structure (Goal: 2026-03-10)

Implement a k-NN-based artist classifier: given a single song's feature vector, return the top-$k$ most probable artists ranked by proximity in feature space. This directly reuses the artist profiling and distance machinery from Milestone 3.

Additionally, define a query language that allows structured artist queries. The most likely format is JSON, as it is extensible for future NLP integration. Example queries:

**JSON query example:**
```json
{
  "type": "similarity",
  "artist": "Radiohead",
  "k": 10,
  "match": ["energy: 0.6", "valence: 0.3"],
  "exclude": {
    "genre": ["pop"],
    "year_range": [[2010, 2026]]
  }
}
```

**Constraint list example:**
```
type = similarity
artist = Radiohead
k = 10
match = energy, valence
exclude_genre = pop
```

### Milestone 5 – Evaluation & Refactoring (Goal: 2026-03-17)

Evaluate the classifier and similarity system against a held-out subset of the Music4All dataset. Metrics will include classification accuracy (top-1 and top-5) for the artist classifier, and qualitative inspection of similarity results for known artist pairs. This milestone will also be used to refactor, clean up the codebase, and address any preprocessing issues identified during development.

### Milestone 6 – Link to LLM (Goal: 2026-03-31)

Integrate the module with the Music Maven chatbot via a **FastAPI** microservice endpoint. Determine how to translate natural language user input (e.g., "Who makes music like this but more upbeat?") into a valid structured JSON query, using the LLM to extract intent and map it to features and weights. Align with the MP2.G1 architecture for service registration and API schema.

### Future Work

Explore richer artist representations beyond feature means — for example, using distribution summaries (histograms, GMMs) or learned embeddings (contrastive learning) for more expressive artist profiles. Considering this approach is being undertaken by group6, there is potential for collaboration here. Investigate graph-based approaches (artist influence networks via Neo4j) to complement acoustic similarity with relational data.

# Tools/Dataset/Libraries

`Music4All Dataset` <br>

`Pandas` <br>

`NumPy` <br>

`Scikit-learn` <br>

`FastAPI` <br>

`JSON` <br>

`LLM API` <br>

# Roles

## Kevin:

### Objective 1: Build artist similarity search engine

- PI.1 (basic): Implement weighted Euclidean distance calculation between artist profiles
- PI.2 (basic): Return ranked list of top-k similar artists given a query artist
- PI.3 (expected): Compare weighted Euclidean distance vs cosine similarity across different feature subsets
- PI.4 (expected): Implement filtering logic to exclude artists by genre, year range, or other metadata constraints
- PI.5 (advanced): Develop query weighting system that allows users to emphasize specific features (e.g., "similar in energy but not valence")

#### Considerations:

- Handle categorical vs continuous data differently
- Weight artist song count differences accordingly for normalization
- Missing/null values in MIR features need imputation or filtering

### Objective 2: Develop k-NN artist classification system

- PI.1 (basic): Implement basic k-NN classifier that returns top-k artists given a song's feature vector
- PI.2 (basic): Calculate weighted Euclidean distance between song features and artist profiles
- PI.3 (expected): Evaluate classifier accuracy using top-1 and top-5 metrics on held-out test set
- PI.4 (expected): Generate confusion matrices showing which artists are most commonly misclassified as each other
- PI.5 (advanced): Optimize k parameter and feature weights through cross-validation to maximize classification accuracy

#### Considerations:

- Determine appropriate k-value
- Distance metrics sensitive to feature scales - normalization is necessary
- Dataset imbalances

## Liam:

### Objective 1: Implement/refine artist profiling system from MIR features

- PI.1 (basic): Load and parse the Music4All dataset to extract per-song MIR features (tempo, energy, valence, danceability, key, mode)
- PI.3 (expected): Normalize feature vectors to comparable scales and store artist profiles in structured Pandas DataFrames
- PI.4 (expected): Compare different aggregation methods (mean vs median vs variance) and evaluate their impact on artist representation
- PI.5 (advanced): Implement feature selection to identify which MIR features are most discriminative for artist classification

#### Considerations:

- Cosine vs Euclidean - pros/cons
- Feature weight (via input) summation should = 1, or be normalized
- Balance metadata filter constraints

### Objective 2: Integrate Artist Similarity System with Music Maven chatbot via API

- PI.1 (basic): Set up FastAPI microservice with endpoints for similarity and classification queries
- PI.2 (basic): Define JSON schema for structured queries including artist, k, feature weights, and exclusion filters
- PI.3 (expected): Implement query parser that validates and processes JSON requests into system parameters
- PI.4 (expected): Connect API endpoints to the LLM chatbot for natural language query translation
- PI.5 (advanced): Design and test LLM prompt that translates conversational queries (e.g., "like Radiohead but more upbeat") into valid JSON structure

#### Considerations:

- Natural Language ambiguity
- AI hallucination
- Discrete query documentation





# Bibliography

**Artist Identification & Classification**

- Whitman, B., & Lawrence, S. (2002). "Inferring descriptions and similarity for music from community metadata." *Proceedings of the 2002 International Computer Music Conference (ICMC)*.
- Bergstra, J., Casagrande, N., Erhan, D., Eck, D., & Kégl, B. (2006). "Aggregate features and AdaBoost for music classification." *Machine Learning*, 65(2-3), 473–484.
- Mandel, M., & Ellis, D. (2005). "Song-level features and support vector machines for music classification." *Proceedings of the 6th International Conference on Music Information Retrieval (ISMIR)*.

**Artist Similarity**

- Seyerlehner, K., Schedl, M., Pohle, T., & Knees, P. (2010). "Using block-level features for genre classification, tag classification, and music similarity estimation." *Proceedings of MIREX*.
- Pampalk, E., Flexer, A., & Widmer, G. (2005). "Improvements of audio-based music similarity and genre classification." *Proceedings of the 6th International Conference on Music Information Retrieval (ISMIR)*.
- Schedl, M., Flexer, A., & Urbano, J. (2013). "The neglected user in music information retrieval research." *Journal of Intelligent Information Systems*, 41(3), 523–539.

**MIR Features & Similarity Measures**

- Tzanetakis, G., & Cook, P. (2002). "Musical genre classification of audio signals." *IEEE Transactions on Speech and Audio Processing*, 10(5), 293–302.
- https://www.researchgate.net/publication/224224057_Unifying_Low-Level_and_High-Level_Music_Similarity_Measures
- https://ismir-explorer.ai.ovgu.de/app/view/pdf/1787/
- https://ismir-explorer.ai.ovgu.de/app/view/pdf/1882/

**K-Nearest Neighbours & Classification**

- https://ieeexplore-ieee-org.ezproxy.library.uvic.ca/document/4457263
- https://ismir-explorer.ai.ovgu.de/app/view/pdf/983/

**NLP / LLM Integration**

- https://www.ijcai.org/proceedings/2024/0841.pdf
- https://arxiv.org/pdf/2503.24193

