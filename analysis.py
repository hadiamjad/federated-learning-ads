import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

def createMatrix():
    ratings = pd.read_csv(r'data/overall_ad_ratings.csv')
    ads_categories = pd.read_csv(r'data/content_labels.csv')
    # join the two dataframes on pid
    df = pd.merge(ratings, ads_categories, on='ad_id')
    # drop columns that are not needed
    df = df.drop(columns=['free_response'])
    
    # user-item matrix
    matrix = {}
    categories = ['Advertorial', 'Apparel', 'B2B Products', 'Banking/Credit Cards',               'Beauty Products', 'COVID Products', 'Cars', 'Celebrity News',               'Cell Service', 'Computer Security-related', 'Consumer Tech',               'Contest', 'Dating', 'Decoy', 'Education', 'Employment',               'Entertainment', 'Food and Drink', 'Games and Toys', 'Genealogy',               'Gifts', 'Google Responsive', 'Health and Supplements',               'Household Products', 'Human Interest', 'Humanitarian', 'Image',               'Insurance', 'Investment Pitch', 'Journalism', 'Legal Services',               'Listicle', 'Medical Services and Prescriptions', 'Mortgages', 'Native', 'Pets', 'Political Campaign', 'Political Content', 'Political Memorabilia', 'Political Poll', 'Poll', 'Public Records Service', 'Public Relations', 'Real Estate', 'Recreational Drugs', 'Religious', 'Scientific Journal', 'Self-Link', 'Senior Living', 'Social Media','Software Download', 'Sponsored Content', 'Sponsored Search', 'Sports','Travel', 'Weapons', 'Wedding Services']

    for i in df.index:
        # add user in matrix if not already there
        if df['pid'][i] not in matrix:
            matrix[df['pid'][i]] = {category: 0 for category in categories}
        for category in categories:
            if df[category][i] == 1:
                matrix[df['pid'][i]][category] = max(df['overall_rating'][i],matrix[df['pid'][i]][category]) 
        # # if user disliked the ad, add 0 to the category
        # elif df['overall_rating'][i] < 5:
        #     for category in categories:
        #         if df[category][i] == 1:
        #             matrix[df['pid'][i]][category] = df['overall_rating'][i],matrix[df['pid'][i]][category])
    arr =  np.array([[v.get(cat, 0) for cat in categories] for k, v in matrix.items()])
    return arr, matrix

def createGoogleVisionMatrix():
    # read the dictionary from the JSON file
    with open('my_dict.json', 'r') as file:
        my_dict = json.load(file)
    
    categories = []
    for key in my_dict:
        for category in my_dict[key]:
            if category[0] not in categories:
                categories.append(category[0])
    print(len(categories))
    # user-item matrix
    matrix = {}
    df = pd.read_csv(r'data/overall_ad_ratings.csv')
    for i in tqdm(df.index, desc='Processing files'):
        # add user in matrix if not already there
        if df['pid'][i] not in matrix:
            matrix[df['pid'][i]] = {category: 0 for category in categories}
        for category in my_dict[str(df['ad_id'][i])+".webp"]:
            matrix[df['pid'][i]][category[0]] = max(df['overall_rating'][i], matrix[df['pid'][i]][category[0]])
            
    arr =  np.array([[v.get(cat, 0) for cat in categories] for k, v in matrix.items()])
    return arr, matrix

def train_test_split(matrix, test_size=0.2):
    """
    Splits the user-item matrix into training and test sets.
    """
    num_users, num_items = matrix.shape
    test = np.zeros((num_users, num_items))
    train = matrix.copy()
    for user in range(num_users):
        non_zero_ratings = matrix[user, :].nonzero()[0]
        test_ratings = np.random.choice(non_zero_ratings,
                                         size=int(test_size * len(non_zero_ratings)),
                                         replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = matrix[user, test_ratings]
    assert(np.all((train * test) == 0))
    return train, test

def KNerestNeighbors(train, test):
    # Set the number of latent factors
    k = 2

    # Initialize the user and item matrices with random values
    U = np.random.rand(train.shape[0], k)
    V = np.random.rand(train.shape[1], k)

    # Set the learning rate and regularization parameter
    alpha = 0.1
    lambd = 0.5

    # Define the number of iterations to perform
    num_iters = 100000

    # Perform stochastic gradient descent to optimize the matrices
    for iteration in range(num_iters):
        for i in range(train.shape[0]):
            for j in range(train.shape[1]):
                if train[i,j] > 0:
                    error = train[i,j] - np.dot(U[i,:], V[j,:].T)
                    U[i,:] += alpha * (error * V[j,:] - lambd * U[i,:])
                    V[j,:] += alpha * (error * U[i,:] - lambd * V[j,:])

    # Predict the missing ratings
    predicted_R = np.dot(U, V.T)

    # Compute the RMSE on the test set
    mse = np.sum((predicted_R - test) ** 2)
    num_ratings = np.sum(test > 0)
    rmse = np.sqrt(mse / num_ratings)
    print("RMSE:", rmse)

def random_forest(train, test):
    # Train a random forest model
    model = RandomForestRegressor(n_estimators=100000, max_depth=10, min_samples_split=2, random_state=42)

    # Flatten the training data into a 2D array
    X_train = train.reshape((-1, train.shape[-1]))
    y_train = X_train[:, -1]
    X_train = X_train[:, :-1]

    # Flatten the test data into a 2D array
    X_test = test.reshape((-1, test.shape[-1]))
    y_test = X_test[:, -1]
    X_test = X_test[:, :-1]

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the ratings on the test set
    predicted_R = model.predict(X_test)

    # Compute the RMSE on the test set
    mse = np.sum((predicted_R - y_test) ** 2)
    num_ratings = np.sum(y_test > 0)
    rmse = np.sqrt(mse / num_ratings)
    print("RMSE:", rmse)

    # Compute the MAE on the test set
    mae = np.mean(np.abs(predicted_R - y_test))
    print("MAE:", mae)

    # Compute the accuracy on the test set
    accuracy = accuracy_score(y_test, predicted_R)
    print("Accuracy:", accuracy)


def main():
    arr, matrix = createGoogleVisionMatrix()
    # arr, matrix = createMatrix()
    train, test = train_test_split(arr)
    random_forest(train, test)
main()