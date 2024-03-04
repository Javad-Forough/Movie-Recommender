# Movie Recommender System

This is a movie recommender system implemented in Python using machine learning techniques, specifically neural networks. The system utilizes the TMDb dataset to recommend movies based on user preferences and movie similarities.

## Features

- Loads and preprocesses movie data from TMDb dataset.
- Extracts features from movie titles and overviews using TF-IDF vectorization.
- Utilizes neural network model to learn representations of movie features.
- Recommends similar movies based on learned representations.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/movie-recommender.git
    cd movie-recommender
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script:**

    ```bash
    python main.py
    ```

## Dataset

The TMDb dataset contains various attributes of movies such as title, overview, original language, vote count, and vote average. This dataset is used to train the recommender system.

## Usage

The main script loads the dataset, preprocesses the data, trains the neural network model, and provides recommendations based on user input.


