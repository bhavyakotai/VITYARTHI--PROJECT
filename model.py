import csv
import math
import os

# --- 1. DATASET GENERATION ---
def create_dataset():
    data_dir = '../data'
    file_path = os.path.join(data_dir, 'dataset.csv')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Header: Sleep, Screen, Study, Assign, Social, Stress, Target (1=Procrastinate)
    rows = [
        ['sleep', 'screen', 'study', 'assign', 'social', 'stress', 'target'],
        [8, 2, 7, 1, 1, 2, 0], [4, 9, 1, 5, 7, 9, 1],
        [7, 3, 6, 2, 2, 3, 0], [5, 7, 2, 4, 6, 8, 1],
        [6, 4, 5, 3, 3, 5, 0], [4, 10, 0, 6, 8, 10, 1],
        [9, 1, 8, 0, 1, 1, 0], [5, 8, 2, 5, 5, 7, 1],
        [7, 2, 5, 1, 2, 3, 0], [3, 8, 1, 6, 7, 9, 1],
        [8, 3, 6, 2, 1, 2, 0], [4, 7, 2, 5, 4, 8, 1],
        [6, 5, 4, 3, 4, 6, 0], [5, 9, 1, 4, 8, 9, 1],
        [7, 4, 5, 2, 3, 4, 0]
    ]

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"✅ Dataset initialized at {file_path}")

# --- 2. CORE ML LOGIC (From Scratch) ---
def sigmoid(z):
    # Clips z to avoid overflow errors in math.exp
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))

def predict_probability(features, weights):
    # z = bias + (w1*x1 + w2*x2...)
    z = weights[0] 
    for i in range(len(features)):
        z += weights[i + 1] * features[i]
    return sigmoid(z)

def train_model(X, y, lr=0.01, epochs=1500):
    # Weights initialized to 0: [bias, w1, w2, w3, w4, w5, w6]
    weights = [0.0] * (len(X[0]) + 1)
    
    for _ in range(epochs):
        for i in range(len(X)):
            prediction = predict_probability(X[i], weights)
            error = y[i] - prediction
            
            # Gradient Descent update rule
            # weight = weight + learning_rate * error * gradient
            weights[0] += lr * error * prediction * (1 - prediction)
            for j in range(len(X[i])):
                weights[j+1] += lr * error * prediction * (1 - prediction) * X[i][j]
    return weights

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    create_dataset()

    # Load data
    X, y = [], []
    with open('../data/dataset.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader) # Skip headers
        for row in reader:
            X.append([float(val) for val in row[:-1]])
            y.append(float(row[-1]))

    print("🤖 Training Procrastination Model...")
    model_weights = train_model(X, y)
    print("✨ Training Complete!\n")

    # Interactive UI
    print("--- Enter Your Current Stats ---")
    user_data = [
        float(input("Sleep hours: ")),
        float(input("Screen time (hrs): ")),
        float(input("Study hours: ")),
        float(input("Assignments due: ")),
        float(input("Social media time (hrs): ")),
        float(input("Stress level (1-10): "))
    ]

    prob = predict_probability(user_data, model_weights)

    print(f"\nResult: {prob:.2%} risk of procrastination.")
    if prob >= 0.5:
        print(" ACTION REQUIRED: You are likely to procrastinate.")
        print("Tip: Use the '5-Minute Rule'—commit to working for just 5 minutes.")
    else:
        print("STICK TO IT: You are in a high-productivity state!")