import numpy as np
import pandas as pd
import json
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics import precision_score, recall_score
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

def KnnModelPrecisionRecall(df):
    print(df.shape)
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 2))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Define the algorithm and run cross-validation
    algo = KNNBasic()
    # Train the algorithm on the training set
    algo.fit(trainset)

    # Test the algorithm on the testing set
    predictions = algo.test(testset)

    # Convert the predictions to binary labels using the threshold of 1.5
    true_labels = [1 if true_rating <= 1.5 else 2 for (_, _, true_rating) in testset]
    predicted_labels = [1 if pred_rating[3] <= 1.5 else 2 for pred_rating in predictions]


    # Compute precision and recall scores
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    print("Precision:", precision)
    print("Recall:", recall)
   
def SVDModel(df):
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 2))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    # Define the algorithm and run cross-validation
    algo = SVD()
    results = cross_validate(algo, data, measures=['mae'], cv=5, verbose=True)

    algo.fit(data.build_full_trainset())
    pred = algo.test([(26, "Water", 1)])
    print(pred)
    # Print the mean RMSE across all folds
    print("Mean MAE:", results['test_mae'].mean())

def SVDModelPrecisionRecall(df):
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 2))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Define the algorithm and run cross-validation
    algo = SVD()

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Test the algorithm on the testing set
    predictions = algo.test(testset)

    # Convert the predictions to binary labels using the threshold of 1.5
    true_labels = [1 if true_rating <= 1.5 else 2 for (_, _, true_rating) in testset]
    predicted_labels = [1 if pred_rating[3] <= 1.5 else 2 for pred_rating in predictions]


    # Compute precision and recall scores
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    print("Precision:", precision)
    print("Recall:", recall)

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

def torchModelPrecisionRecall(df):
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
    precision_scores = []
    recall_scores = []

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

        # Compute precision and recall
        binary_val_preds = [1 if pred <= 1.5 else 2 for pred in val_preds]
        precision_score_val = precision_score(val_df["rating"], binary_val_preds, pos_label=1)
        recall_score_val = recall_score(val_df["rating"], binary_val_preds, pos_label=1)
        precision_scores.append(precision_score_val)
        recall_scores.append(recall_score_val)
        print(f"Precision: {precision_score_val:.4f}, Recall: {recall_score_val:.4f}")

    # Calculate the mean and standard deviation of the MAE, precision, and recall scores
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    print(mean_precision)
    print(mean_recall)

def venn(df):
    # Create a list of sets
    sets = [
        set(df[df["rating"] == 1]["item_id"]),
        set(df[df["rating"] == 2]["item_id"]),
    ]

    # Using the & operator
    intersection_set = sets[0] & sets[1]
    print(len(sets[1]-intersection_set))


def main():
    matrix = createGoogleVisionMatrix()
    df = createtSupriseData(matrix)
    print(df.head(5))
    torchModelPrecisionRecall(df)
 
main()