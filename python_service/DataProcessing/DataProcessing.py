# pyspark_data_processing.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count

def processMovieData():
    # Initialize Spark session
    spark = SparkSession.builder.appName("MovieRec").getOrCreate()

    # Load the movies and ratings data (MovieLens-style, delimited by "::")
    movies_df = spark.read.csv("data/movies.dat", sep="::", inferSchema=True).toDF("MovieID", "Title", "Genres")

    return movies_df

def processUserData():
    spark = SparkSession.builder.appName("UserRec").getOrCreate()
    users_df = spark.read.csv("data/users.dat", sep="::", inferSchema=True).toDF("UserID", "Gender", "Age", "Occupation", "Zipcode")
    ratings_df = spark.read.csv("data/ratings.dat", sep="::", inferSchema=True).toDF("UserID", "MovieID", "Rating", "Timestamp")
    ratings_with_users = ratings_df.join(users_df, "UserID")

    return users_df, ratings_with_users