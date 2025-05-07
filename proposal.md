# Capstone: Universal Behavioral Profile Generation for a Unified Approach to Multi-Task Behavioral Prediction

## Summary

This project aims to address the "Universal Behavioral Modeling Data Challenge" by developing Universal Behavioral Profiles from user interaction logs.

These profiles are user representations designed to encode essential aspects of individual past interactions [e.g., purchases, add to cart, page visits, search queries].

The core goal is to create profiles that are universally applicable and can generalize effectively across multiple predictive tasks, including churn prediction, category propensity, and product propensity, as well as undisclosed hidden tasks.

The project involves analyzing a large, anonymized dataset of real-world user interaction logs, creating user embeddings (profiles), and submitting these for evaluation via a fixed neural network architecture provided by the challenge organizers.

The final performance will be an aggregate of results from all tasks, emphasizing the generalizability of the created profiles.

## Project Proposal

### Business Problem

**Who might benefit from this project? How?**

* Modern enterprises relying on machine learning and predictive analytics for business decisions. This includes areas like e-commerce, streaming services, and any domain requiring understanding and prediction of user behavior.
* They would benefit by having a unified modeling approach, where a single set of user representations can power various predictive tasks (recommendation, propensity, churn, LTV prediction), rather than developing separate models for each. This leads to more efficient model development and potentially more robust and generalizable predictions.

**What is the goal?**

* To develop Universal Behavioral Profiles based on the provided user interaction data.
* These profiles (user embeddings) should be effective inputs for a simple neural network to perform well on several predictive tasks [churn, category propensity, product propensity, and hidden tasks].
* The submitted embeddings must be a dense matrix with a maximum embedding dimension of 2048 and `dtype=float16`.

### Approach

**Data Description (Initial EDA):**

* Source of the data: Anonymized dataset of real-world user interaction logs provided by the "Universal Behavioral Modeling Data Challenge" organizers. This includes product properties.
* What it describes: Each row in the event files (`product_buy`, `add_to_cart`, `remove_from_cart`, `page_visit`, `search_query`) represents a specific user interaction with associated `client_id`, `timestamp`, and event-specific details (e.g., `sku`, `url`, `query`). `product_properties` describes items by `sku`, `category`, `price` (bucketed), and `name` [quantized embedding].
* Shape of the data
  * `product_buy`: 1,682,296 events
  * `add_to_cart`: 5,235,882 events
  * `remove_from_cart`: 1,697,891 events
  * `page_visit`: 150,713,186 events
  * `search_query`: 9,571,258 events
  * Profiles need to be generated for 1,000,000 users specified in `relevant_clients.npy`.
* Initial assessment: Analyze column types (int64, object for timestamps and quantized vectors) and check for nulls within the Parquet files. Understand the encoding for text ('name', 'query' as 16-number quantized embeddings) and decimal ('price' as 100 quantile-based buckets) columns.

**Data Cleaning & Feature Engineering (Focus on creating embeddings):**

* Timestamps: Convert to numerical features (e.g., time since last event, day of week, hour of day).
* Categorical IDs (`sku`, `category`, `url`): Consider how to represent these numerically if not directly using them for sequence embedding models.
* Quantized Embeddings (`name`, `query`): These are already vector representations; decide how to aggregate or utilize them in the user profile (e.g., averaging, using in a sequence model).
* Develop strategies to aggregate user interaction history into a fixed-size vector (Universal Behavioral Profile) up to 2048 dimensions. This could involve:
  * Recency-weighted aggregations.
  * Sequence embedding models (e.g., RNNs, Transformers) applied to event sequences per user.
  * Counts/frequencies of different event types or item interactions.
  * Graph-based features if user-item interactions can be modeled as a graph.

**Data Preparation (for local testing, using provided pipeline):**

  * Use the provided `data_utils/split_data.py` script to split data for internal testing into input events, train target events, and validation target events. **Note:** For final submission, embeddings must be created using ALL provided events.
  * The primary output is the `embeddings.npy` and `client_ids.npy` files.

**Modeling and Model Evaluation (Focus on generating embeddings, downstream tasks handled by organizers):**

* The core task is to generate the Universal Behavioral Profiles (embeddings).
* Local evaluation can be performed using the provided competition code (`training_pipeline.train`) which trains a fixed neural network architecture on the generated embeddings for the open tasks (`churn`, `propensity_category`, `propensity_sku`).
* The model architecture is fixed: three Inverted Bottleneck blocks.
* Performance metrics: AUROC is primary. For propensity tasks, a weighted sum of AUROC, Novelty, and Diversity is used [0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity].
* The final leaderboard score aggregates ranks using Borda count.

### Outcomes

**Model Tuning (of the embedding generation process):**

* This involves tuning the parameters of the chosen embedding generation technique (e.g., architecture of sequence models, aggregation methods, feature selection).
* Iterate based on local evaluation performance on open tasks, keeping in mind the need for generalization to hidden tasks.

**Model Selection (of the best embedding generation strategy):**

* Select the embedding generation strategy that yields the best performance on local validation and seems most likely to generalize.

**Submission:**

* Prepare `client_ids.npy` (1D NumPy ndarray, int64, matching `relevant_clients.npy` order) and `embeddings.npy` [2D NumPy ndarray, float16, max dimension 2048].
* Validate the entry format using `universal_behavioral_modeling_challenge/validator/run.py`.
* Submit these two files.

### Follow-up

**How will this project help me further my research/career?**
* This project provides hands-on experience with large-scale behavioral data, advanced representation learning techniques, and participation in a competitive data science challenge, all of which are highly valuable for a career in machine learning and data science, particularly in areas like recommender systems, user modeling, and online advertising. The focus on generalizable representations is at the forefront of current research. It also lines up with my work history in marketing.

**How will you communicate results with the research community, as well as the public?**
* Through the final class presentations on May 22nd and a public GitHub repository containing the code for embedding generation, documentation, and analysis. Also a README detailing the approach and learnings from the challenge.
