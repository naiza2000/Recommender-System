# Recommender-System
I have implemented the collaborative filtering learning algorithm and applied it to a dataset of movie ratings*. This dataset consists of ratings on a scale of 1 to 5. The dataset has  users, and  movies. I haveimplemented the function cofiCostFunc.m that computes the collaborative fitlering objective function and gradient. After implementing the cost function and gradient, I have used fmincg.m to learn the parameters for collaborative filtering.
</br>(*MovieLens 100k Dataset)[https://grouplens.org/datasets/movielens/] from GroupLens Research.
</br>
</br>
The code in main.py will load the dataset ex8_movies.mat, providing the variables Y and R in MATLAB environment. The matrix Y (a num_movies  num_users matrix) stores the ratings  (from 1 to 5). The matrix R is an binary-valued indicator matrix, where R(i,j) = 1 if user j gave a rating to movie i, and R(i,j) = 0 otherwise. The objective of collaborative filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries with R(i,j) = 0. This will allow us to recommend the movies with the highest predicted ratings to the user.
</br>
</br>
Y is a 1682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users
</br>R is a 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
</br>

