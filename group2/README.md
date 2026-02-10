# Proposed Project 
For this subsection of ‘Music Maven’, we propose the creation of a steerable music recommender used to curate playlists based upon user specification. To achieve this, we seek to augment a basic content-based recommender through the addition of a querying language utilizing a set of primitive constraints. In turn, enabling a further refinement of the recommendation process to ensure that generated playlists are uniquely tailored to a user's desires. 

The high-level usage of the system will consist of a user creating a structured query in order to generate a playlist of recommended songs, for which varying fields will be specified. The first of these will be the size of the playlist to be generated, creating in turn a playlist of size ‘k’ songs ranked by similarity. Following this, a ‘seed’ song will then be specified, this will be used to provide the basis for content-based recommendation by providing a set of ‘features’ (i.e tempo, pitch, bpm) for use of similarity measures against other songs. Next, a user will then be able to specify a set of constraints to be imposed for song selection. This will allow for songs that do not abide by the specified constraints to be filtered and excluded before direct comparison occurs (i.e. exclude all songs whose genre is jazz). Lastly, a user will also have the ability to specify which features of a song will be used for direct comparison via similarity measures. Enabling direct comparison to only consider a certain subset of features to determine relative similarly. In conjunction, this will allow for a high-level query such as “Given Paranoid Android by Radiohead, create a playlist of 10 songs comparing only tempo and key but not belonging to genre ‘Pop’” to be represented as a structured query.

In order to facilitate this, implementation of this system will be broken into three general categories, being that of: 1.) Creation of a query language to handle representation of user queries, 2.) Creation of a system to utilize provided constraints to conduct database operations for filtering, and  3.) Comparison of songs within resulting search space through usage of similarity methods. 

The first of these will likely be handled through use of a JSON query with fields corresponding to: 1.) size of playlist to be created, 2.) ‘seed’ song to be used, 3.) inclusionary and exclusionary constraints, and 4.) features to be used for comparison. Research will need to be done on the implementation of JSON for this usage context.

For database operations, following research of the scheme used by the dataset ‘Music4All’, a framework such as ‘Pandas’ will likely be used to store associated data. Allowing for database operations akin to ‘projection’ and ‘section’ to be used for filtering based on associated metadata. 

Lastly, research will need to be conducted on similarity measures that can be used within the resulting search space such as ‘k-nearest neighbours’ to rank songs that are most similar to the provided ‘seed’ song. As the system seeks to enable for comparison on only selected features, further research will need to be done for methods such as a weighted approach for feature importance. 

Following initial implementation, we propose that this system provides the needed foundation for integration with the additional subsections, providing a robust and effective recommender. 

# Timeline & Objectives
Milestone 1 – Import Dataset, Initialize Repositories (Due 2026-02-17)
Music Maven will be using the music4all (M4A) dataset. The first goal is to obtain a copy of the dataset – either locally or shared. Ideally, we will obtain a copy of just the dataset labels (without audio) for local testing and training. We will also create branches from the main and ensure we are able to safely push and pull.

Milestone 2 – Preprocessing & Basic Queries (Due 2026-02-24)
Ensure the M4A dataset is adequate for a similarity search. That is, removing unneeded features, scaling numeric values and encoding categories. The pruned and processed dataset will be used for any training and querying. We can also begin to query the dataset using basic Pandas Dataframe functions.

MIR Features in M4A
Dancability
Energy
Key
Mode
Valence

	Metadata Features in M4A
ID
Album Name
Release Date
Genres

Milestone 3 – Similarity Search (Due 2026-03-03)
Given a single seed song tuple, write an algorithm that uses a Weighted Euclidean Search to determine which other songs in the dataset contain the most similar numeric attributes. The formula for Weighted Euclidean Search is given by:

$$
d(\mathbf{x}, \mathbf{y}) =
\sqrt{
\sum_{i=1}^{N}
w_i \left(x_i - y_i\right)^2
}
$$

Milestone 4 – Query Structure, K-Nearest Songs (Due 2026-03-10)
Define a query language that allows the user to set constraints. This will most likely be done using a JSON query, as it will be the most extensible for NLP and linking to an LLM in the future. Other possible query structures include a Boolean Query, an EBNF Grammar, or a basic constraint list. 

JSON query:
{“k”: 100,
“Seed”: “songs/seed.wav”,
“Match”: [“valence: 0.7”, “tempo : 0.2”],
“Exclude”: {
	“Genre”: [“jazz”],
“Year_range”: [[1945, 1970]]}

Constraint List:
k = 100
match = valence, tempo
	exclude_genre = jazz
	exclude_year = 1945-1970

Boolean Query:
	User: k=100, (valence & tempo) & !year:1945-1970 & !genre:jazz

Milestone 5 – Evaluation & Refactoring (Due 2026-03-17)
This time will be used to evaluate our current progress, iterate, refactor, and test. If time permits, begin exploring how to link for Natural Language Processing and converting user input to JSON structured query.

Milestone 6 – Link to LLM (Due 2026-03-31)
Determine how to integrate our work into the Music Maven – ideally with NLP. Have the LLM process human input and return a structured JSON query. Map keywords such as “vibe” to features such as Dancibility, Energy and Valence, and favour their weights in the search.

Future Work
Explore API options to generate this playlist in a usable space – Youtube, Spotify, etc.

# Tools/Dataset/Libraries
Music4All Dataset
Found  here: music4all
Suggested and shared by George Tzanetakis.

Pandas
We will use Pandas Dataframes to hold the features from the M4a dataset. This allows us to do basic projections, selections, and preprocessing. 

Numpy
Used for vectorization, normalization, and search.

JSON
Queries will be stored in JSON files following a specific template.

Scikit-learn
Potential use for clustering, training and processing.

LLM API
A way to allow users to input natural language.

# Roles
Thomas Pietrovito:
Dataframe management
Preprocessing
Weighted Euclidean Search
Link to NLP and LLM

Noah Hicks:
Reading / Writing to JSON files
Query parsing, structure, schema
Extracting / Inferring weights 
Link to NLP and LLM

# Progress Indicators
## Thomas Pietrovito
PI.1 (Basic): Load M4A feature and metadata into a structured pandas Dataframe
PI.2 (Basic): Normalize data and preprocess dataset for searching
PI.3 (Expected): Implement a Euclidean search algorithm that returns a desired number of closest matches
PI.4 (Expected): Extend search algorithm to accept weights and feature importance
PI.5 (Advanced): Prototype an interface that allows features and weights to be offered from natural language input
## Noah Hicks
PI.1 (Basic): Implement reading/writing to JSON files
PI.2 (Basic): Map JSON query to pandas dataset filtering functions
PI.3 (Expected): Extend input to accept inferred weights (e.g, “faster”, more energetic”)
PI.4 (Expected): Prototype an NLP pipeline (via spaCy or LLM API) that converts natural language into valid JSON query
PI.5 (Advanced): Explore failure and ambiguity in natural queries, local tasks vs generative tasks

# Literature

# Bibliography
Grouping / Clustering

chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/1956/
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/983/
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/830/
https://ieeexplore-ieee-org.ezproxy.library.uvic.ca/document/4457263

Constraints
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/478/

Playlists
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/1830/
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/1372/
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ismir-explorer.ai.ovgu.de/app/view/pdf/461/

NLP / CSP / LLM
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ijcai.org/proceedings/2024/0841.pdf 
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2503.24193


Features, Metadata, and Dimensionality Reduction
https://ismir2011.ismir.net/papers/PS1-10.pdf
https://www.researchgate.net/publication/224224057_Unifying_Low-Level_and_High-Level_Music_Similarity_Measures
https://ismir-explorer.ai.ovgu.de/app/view/pdf/1882/
https://ismir-explorer.ai.ovgu.de/app/view/pdf/1787/

