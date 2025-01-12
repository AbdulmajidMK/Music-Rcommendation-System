from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

app = Flask(__name__)

def load_data():
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10],
        'age': [20, 23, 25, 26, 29, 30, 31, 33, 37, 20, 21, 25, 26, 27, 30, 31, 34, 35, 20, 25],
        'gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
                   'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female'],
        'genre': ['HipHop', 'HipHop', 'HipHop', 'Jazz', 'Jazz', 'Jazz', 'Classical', 'Classical', 'Classical',
                  'Dance', 'Dance', 'Dance', 'Acoustic', 'Acoustic', 'Acoustic', 'Classical', 'Classical', 'Classical', 'Jazz', 'Dance'],
        'rating': [5, 4, 3, 4, 5, 3, 5, 4, 3, 4, 5, 2, 3, 5, 4, 3, 5, 4, 5, 2]
    }
    return pd.DataFrame(data)

@app.route('/load_data', methods=['GET'])
def get_data():
    data = load_data()
    return jsonify(data.to_dict(orient='records'))

@app.route('/add_user', methods=['POST'])
def add_user():
    data = load_data()
    new_entry = request.json
    new_row = pd.DataFrame(new_entry, index=[0])
    updated_data = pd.concat([data, new_row], ignore_index=True)
    return jsonify(updated_data.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = load_data()
    user_item_matrix = data.pivot_table(index='user_id', columns='genre', values='rating', aggfunc='mean').fillna(0)
    similarity = cosine_similarity(user_item_matrix)
    predictions = predict_ratings(similarity, user_item_matrix)

    request_data = request.json
    user_id = request_data.get('user_id')
    n = request_data.get('n', 5)

    user_index = user_id - 1  # Zero-indexed
    sorted_indices = predictions[user_index].argsort()[::-1]
    unseen_items = np.where(user_item_matrix.iloc[user_index] == 0)[0]
    recommendations = [idx for idx in sorted_indices if idx in unseen_items][:n]
    genres = user_item_matrix.columns[recommendations].tolist()

    return jsonify({'recommendations': genres})

def predict_ratings(similarity, user_item_matrix):
    mean_user_rating = user_item_matrix.mean(axis=1).values.reshape(-1, 1)
    ratings_diff = user_item_matrix - mean_user_rating
    pred = mean_user_rating + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

if __name__ == '__main__':
    app.run(debug=True)
