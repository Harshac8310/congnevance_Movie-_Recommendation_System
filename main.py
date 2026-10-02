import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load Dataset
# -----------------------------

movies = pd.read_csv("dataset/ml-latest-small/movies.csv")
ratings = pd.read_csv("dataset/ml-latest-small/ratings.csv")

data = ratings.merge(movies, on="movieId")

os.makedirs("visualizations", exist_ok=True)


# -----------------------------
# Data Visualizations
# -----------------------------

# Movies by Genre
genre_counts = (
    movies["genres"]
    .str.split("|")
    .explode()
    .value_counts()
)

genre_counts.plot(kind="bar", figsize=(10, 5))

plt.title("Number of Movies by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/movies_by_genre.png")
plt.close()


# Rating Distribution
rating_counts = ratings["rating"].value_counts().sort_index()

rating_counts.plot(kind="bar")

plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.savefig("visualizations/rating_distribution.png")
plt.close()


# Most-Rated Movies
popular_movies = data["title"].value_counts().head(10)

popular_movies.plot(kind="bar", figsize=(12, 6))

plt.title("Top 10 Most-Rated Movies")
plt.xlabel("Movie")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("visualizations/most_rated_movies.png")
plt.close()


# -----------------------------
# User-Movie Matrix
# -----------------------------

user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

# Replace missing ratings with 0 temporarily
# for cosine similarity calculation.
user_movie_matrix_filled = user_movie_matrix.fillna(0)

user_similarity = cosine_similarity(
    user_movie_matrix_filled
)


# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_movies(user_id, n=10):

    # Check whether the user exists
    if user_id not in user_movie_matrix.index:
        print("User ID not found.")
        return pd.DataFrame()

    # Find the position of the selected user
    user_index = user_movie_matrix.index.get_loc(user_id)

    # Similarity of selected user with all users
    user_similarity_scores = user_similarity[user_index]

    # Get the 5 most similar users
    # [1:6] skips the selected user themselves.
    similar_users = user_similarity_scores.argsort()[::-1][1:6]

    # Get ratings of similar users
    similar_users_ratings = user_movie_matrix_filled.iloc[
        similar_users
    ]

    # Get similarity values of those users
    similarity_weights = user_similarity_scores[similar_users]

    # Apply similarity weights to ratings
    weighted_ratings = similar_users_ratings.mul(
        similarity_weights,
        axis=0
    )

    # Sum weighted ratings
    weighted_sum = weighted_ratings.sum(axis=0)

    # Calculate weighted denominator
    rating_weights = (
        (similar_users_ratings > 0)
        .mul(similarity_weights, axis=0)
        .sum(axis=0)
    )

    # Calculate similarity-weighted average
    weighted_average = weighted_sum / rating_weights

    # Movies already rated by the selected user
    user_ratings = user_movie_matrix.loc[user_id]

    # Movies not yet rated by the selected user
    unrated_movies = user_ratings[
        user_ratings.isna()
    ].index

    # Keep scores only for unrated movies
    recommendation_scores = weighted_average.loc[
        unrated_movies
    ]

    recommendation_scores = recommendation_scores.dropna()

    # Require at least 2 similar users to have rated a movie
    minimum_ratings = 2

    number_of_ratings = (
        similar_users_ratings > 0
    ).sum(axis=0)

    recommendation_scores = recommendation_scores[
        number_of_ratings.loc[
            recommendation_scores.index
        ] >= minimum_ratings
    ]

    # Select top N movies
    top_movies = recommendation_scores.sort_values(
        ascending=False
    ).head(n)

    # Get movie information
    recommended_movies = movies[
        movies["movieId"].isin(top_movies.index)
    ].copy()

    # Add recommendation score
    recommended_movies["score"] = (
        recommended_movies["movieId"].map(top_movies)
    )

    # Sort by recommendation score
    recommended_movies = recommended_movies.sort_values(
        "score",
        ascending=False
    )

    # Round scores for clean presentation
    recommended_movies["score"] = (
        recommended_movies["score"].round(2)
    )

    return recommended_movies[
        ["movieId", "title", "genres", "score"]
    ]


# -----------------------------
# Run Recommendation System
# -----------------------------

user_id = int(input("Enter user ID: "))

recommendations = recommend_movies(user_id)

if not recommendations.empty:

    print("\nRecommended Movies:\n")
    print(recommendations.to_string(index=False))

    # Save recommendations to CSV
    recommendations.to_csv(
        f"recommendations_user_{user_id}.csv",
        index=False
    )

    print(
        f"\nRecommendations saved to "
        f"recommendations_user_{user_id}.csv"
    )