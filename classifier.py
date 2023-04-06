import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from libreco.data import random_split, DatasetPure
from libreco.algorithms import LightGCN  # pure data, algorithm LightGCN
from libreco.evaluation import evaluate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model

# read the dictionary from the JSON file
with open('my_dict.json', 'r') as file:
    my_dict = json.load(file)

# create a set of unique categories
categories = {category[0] for key in my_dict for category in my_dict[key]}

def update_matrix(group, matrix):
    # create a dictionary of categories and their ratings for the ad
    ad_ratings = defaultdict(int)
    for category in my_dict[str(group['ad_id'].iloc[0])+".webp"]:
        ad_ratings[category[0]] = max(group['overall_rating'])

    # update the user-item matrix
    for pid in group['pid']:
        if pid not in matrix:
            matrix[pid] = {category: 0 for category in categories}
        for category, rating in ad_ratings.items():
            matrix[pid][category] = max(rating, matrix[pid][category])
            if matrix[pid][category] > 3:
                matrix[pid][category] = 1
            elif matrix[pid][category] <= 3 and matrix[pid][category] > 0:
                matrix[pid][category] = 2

def createGoogleVisionMatrix():
    # user-item matrix
    matrix = {}
    df = pd.read_csv(r'data/overall_ad_ratings.csv')
    df.groupby(['pid', 'ad_id']).apply(update_matrix, matrix=matrix)
    return matrix

def createtSupriseData(matrix):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(matrix, orient='index')

    # Reset the index to create a 'user_id' column
    df = df.reset_index()

    # Rename the 'index' column to 'user_id'
    df = df.rename(columns={'index': 'user_id'})

    # Melt the DataFrame to convert the nested dictionaries to rows
    df = pd.melt(df, id_vars='user_id', var_name='item_id', value_name='rating')

    # use value_counts() method to count the number of occurrences of each rating
    counts = df['rating'].value_counts()

    # extract the count of rating 0
    count_0 = counts.get(0, 0)

    print('Number of rows with rating 0:', count_0)

    # Filter the DataFrame to remove rows with a value of 0
    df = df[df['rating'] != 0]

    return df

def createTwoDMatrix(data_dict):
    # Get the list of unique user IDs and category IDs
    user_ids = list(data_dict.keys())
    category_ids = set()
    for user_id in user_ids:
        category_ids.update(data_dict[user_id].keys())
    category_ids = list(category_ids)

    # Create a two-dimensional numpy array with zeros
    user_item_matrix = np.zeros((len(user_ids), len(category_ids)))

    # Fill in the non-zero elements in the numpy array
    for i, user_id in enumerate(user_ids):
        for j, category_id in enumerate(category_ids):
            if category_id in data_dict[user_id]:
                user_item_matrix[i, j] = data_dict[user_id][category_id]

    return user_item_matrix

def KnnModel(df):
    print(df.shape)
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 2))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    # Define the algorithm and run cross-validation
    algo = KNNBasic()
    results = cross_validate(algo, data, measures=['MAE'], cv=5, verbose=True)

    pred = algo.predict(str(24), "Finger")
    print(pred)
    # Print the mean RMSE across all folds
    print("Mean MAE:", results['test_mae'].mean())
   
def SVDModel(df):
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 2))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    # Define the algorithm and run cross-validation
    algo = SVD()
    results = cross_validate(algo, data, measures=['MAE'], cv=5, verbose=True)

    algo.fit(data.build_full_trainset())
    pred = algo.test([(26, "Water", 1)])
    print(pred)
    # Print the mean RMSE across all folds
    print("Mean MAE:", results['test_mae'].mean())

def torchModel(df):
    # Preprocess the data
    df["user_id"] = df["user_id"].astype("category")
    df["item_id"] = df["item_id"].astype("category")

    # Define the number of users and items
    num_users = len(df["user_id"].cat.categories)
    num_items = len(df["item_id"].cat.categories)
    print("Number of users:", num_users)
    print("Number of items:", num_items)

    # Define the deep learning model
    def build_model():
        user_input = Input(shape=(1,))
        user_embedding = Embedding(num_users, 10)(user_input)
        user_flatten = Flatten()(user_embedding)

        item_input = Input(shape=(1,))
        item_embedding = Embedding(num_items, 10)(item_input)
        item_flatten = Flatten()(item_embedding)

        dot_product = Dot(axes=1)([user_flatten, item_flatten])
        output = Dense(1)(dot_product)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(loss="mean_squared_error", optimizer="adam")
        
        return model

    # Define the number of folds for cross-validation
    num_folds = 5

    # Split the data into folds and iterate over them
    kf = KFold(n_splits=num_folds, shuffle=True)
    mae_scores = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(df)):
        print(f"Fold {fold + 1}...")

        # Split the data into training and validation sets
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]

        # Build the model
        model = build_model()

        # Train the model
        model.fit(
            x=[train_df["user_id"].cat.codes.values, train_df["item_id"].cat.codes.values],
            y=train_df["rating"].values,
            batch_size=64,
            epochs=10,
            verbose=0
        )

        # Evaluate the model on the validation set
        val_preds = model.predict(
            x=[val_df["user_id"].cat.codes.values, val_df["item_id"].cat.codes.values],
            batch_size=64
        )
        mae_score = mean_absolute_error(val_df["rating"], val_preds)
        mae_scores.append(mae_score)
        print(f"MAE: {mae_score:.4f}")

    # Calculate the mean and standard deviation of the MAE scores
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)

    print(f"\nMean MAE: {mean_mae:.4f} (+/- {std_mae:.4f})")

    # Predict the rating for a specific user and item id
    user_id = 65
    item_id = "Carmine"
    user_code = df["user_id"].cat.codes[df["user_id"] == user_id].iloc[0]
    item_code = df["item_id"].cat.codes[df["item_id"] == item_id].iloc[0]
    rating_pred = model.predict(x=[[user_code], [item_code]])[0][0]
    print(f"Predicted rating for user {user_id} and item {item_id}: {rating_pred:.2f}")

def librecoModel(df):
    data = df.rename(columns={'user_id': 'user', 'item_id': 'item', 'rating': 'label'})
    print(data.head(5))

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)

    # sample negative items for each record
    train_data.build_negative_samples(data_info)
    eval_data.build_negative_samples(data_info)
    test_data.build_negative_samples(data_info)
    print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

    lightgcn = LightGCN(
        task="ranking",
        data_info=data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=2,
        lr=1e-3,
        batch_size=100,
        num_neg=1,
        device="cuda",
    )
    # monitor metrics on eval data during training
    lightgcn.fit(
        train_data,
        verbose=2,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

    # do final evaluation on test data
    evaluate(
        model=lightgcn,
        data=test_data,
        eval_batch_size=8192,
        k=10,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

    # predict preference of user 2211 to item 110
    print(lightgcn.predict(user=24, item="Finger"))
    # # recommend 7 items for user 2211
    # print(lightgcn.recommend_user(user=24, n_rec=7))

    # # cold-start prediction
    # lightgcn.predict(user="ccc", item="not item", cold_start="average")
    # # cold-start recommendation
    # lightgcn.recommend_user(user="are we good?", n_rec=7, cold_start="popular")


def main():
    matrix = createGoogleVisionMatrix()
    df = createtSupriseData(matrix)

    torchModel(df)
 
main()