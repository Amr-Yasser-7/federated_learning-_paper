import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# Step 1: Generate simulated client datasets
def create_client_data(num_clients=5, num_samples_per_client=100):
    # Simulate data for each client (e.g., linear regression problem)
    client_data = []
    true_weights = np.array([2.0, -3.5, 1.2])  # True weights for the linear model
    noise_scale = 0.1

    for _ in range(num_clients):
        # Generate random features (X) and labels (y)
        X = np.random.rand(num_samples_per_client, 3).astype(np.float32)
        y = np.dot(X, true_weights) + np.random.normal(scale=noise_scale, size=(num_samples_per_client,))
        client_data.append((X, y))
    return client_data

# Step 2: Define a simple model
def create_model():
    model = models.Sequential([
        layers.Dense(1, input_shape=(3,), activation='linear')  # Linear regression model
    ])
    model.compile(optimizer='sgd', loss='mse')
    return model

# Step 3: Federated Averaging Algorithm
def federated_averaging(client_models, global_model):
    # Aggregate the weights from all client models
    averaged_weights = []
    for layer_index in range(len(global_model.get_weights())):
        layer_weights = [client_model.get_weights()[layer_index] for client_model in client_models]
        averaged_weights.append(np.mean(layer_weights, axis=0))
    
    # Update the global model with the averaged weights
    global_model.set_weights(averaged_weights)

# Step 4: Simulate Federated Learning Process
def simulate_federated_learning(num_rounds=5, num_clients=5, num_epochs=5):
    # Create simulated client data
    client_data = create_client_data(num_clients=num_clients)
    
    # Initialize the global model
    global_model = create_model()
    
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        # Train local models on each client's data
        client_models = []
        for i, (X, y) in enumerate(client_data):
            print(f"Training client {i + 1}...")
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())  # Start with global weights
            local_model.fit(X, y, epochs=num_epochs, verbose=0)
            client_models.append(local_model)
        
        # Aggregate local models into the global model
        federated_averaging(client_models, global_model)
    
    return global_model

# Step 5: Run the simulation
if __name__ == "__main__":
    final_model = simulate_federated_learning(num_rounds=5, num_clients=5, num_epochs=5)
    print("Final Global Model Weights:")
    print(final_model.get_weights())