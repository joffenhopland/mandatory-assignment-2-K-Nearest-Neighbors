import pandas as pd
import math

# Iris dataset link
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df_iris = pd.read_csv(
    URL,
    header=None,
    names=["sepal length", "sepal width", "petal length", "petal width", "class"],
)

# convert string class labels to numerical labels
labels, unique = pd.factorize(df_iris["class"])
df_iris["label"] = labels

print(f"df_iris: {df_iris}")
print(f"labels: {labels}")


def euclidean_distance(p, q):
    """
    Calculate the Euclidean distance between two points p and q.
    p, q are lists of coordinates.
    """
    # sum to add up the squared differences of coordinates
    # zip(p, q) combines each corresponding element from p and q together.
    distance = sum((px - qx) ** 2 for px, qx in zip(p, q))

    # Use sqrt to get the square root of the sum of squared differences
    return math.sqrt(distance)


def get_neighbors(train_data, test_point, k):
    """
    Find k nearest neighbors for a given test_point.
    train_data is a DataFrame containing the feature vectors and labels.
    """
    distances = []
    for index, row in train_data.iterrows():
        # Calculate the Euclidean distance between the test_point and the current row of feature values (row[:-1]).
        dist = euclidean_distance(row[:-1], test_point)
        # append the row and its distance to the list distances list
        distances.append((row, dist))
    # Sorting the list by distance
    distances.sort(key=lambda x: x[1])
    # get the k nearest neighbors (k smallest distances).
    neighbors = [x[0] for x in distances[:k]]
    return neighbors


def predict(neighbors):
    """
    Make a prediction based on the class labels of neighbors.
    Takes a list of 'k' nearest neighbors as input.
    """
    class_votes = {}  # initialize empty dict to hold the count of each class label
    for neighbor in neighbors:
        class_label = neighbor[-1]  # Extract the class label from the neighbor
        # Count the class labels
        if class_label in class_votes:
            class_votes[class_label] += 1
        else:
            class_votes[class_label] = 1

    # Use max() function to find the class label with the most counts (votes)
    # the key parameter specifies that we want to find the max based on the votes
    predicted_class = max(class_votes, key=class_votes.get)
    return predicted_class


# Define the number of neighbors to consider (k) and the new data point for which we want to make a prediction
k = 3
new_dp = [7.0, 3.1, 1.3, 0.7]

# Fetch the k nearest neighbors and make prediction for the new data point
neighbors = get_neighbors(df_iris, new_dp, k)
# make a prediction based on these neighbors
prediction = predict(neighbors)

# # Map numerical label back to original string label
predicted_label = unique[prediction]
print(f"The predicted class label for the data point {new_dp} is: {predicted_label}")
