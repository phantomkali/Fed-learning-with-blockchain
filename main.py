import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import json
import os
from web3 import Web3
from datetime import datetime
import pickle

class ModelStorage:
    """
    Handles local storage of model weights and metadata, replacing IPFS functionality.
    Stores model updates in a structured way that maintains versioning and allows for recovery.
    """
    def __init__(self, base_dir="model_updates"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_model_weights(self, weights, epoch):
        """
        Save model weights locally with metadata and return a unique identifier.
        The identifier can be used to recover the weights later if needed.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = f"model_update_{timestamp}_epoch_{epoch}"
        
        # Create a directory for this update
        update_dir = os.path.join(self.base_dir, identifier)
        os.makedirs(update_dir, exist_ok=True)
        
        # Save weights and metadata
        weights_path = os.path.join(update_dir, "weights.pkl")
        metadata_path = os.path.join(update_dir, "metadata.json")
        
        # Save weights using pickle for efficient binary storage
        with open(weights_path, 'wb') as f:
            pickle.dump(weights, f)
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "timestamp": timestamp,
            "identifier": identifier
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return identifier

    def load_model_weights(self, identifier):
        """Recover model weights using their identifier"""
        weights_path = os.path.join(self.base_dir, identifier, "weights.pkl")
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                return pickle.load(f)
        return None

class Client:
    def __init__(self, name, features):
        self.name = name
        self.features = features
        self.selected_features = features
        self.model = self._build_model(len(features))
        self.previous_weights = None
        
    def _build_model(self, input_dim):
        """Build neural network model with proper input dimensionality"""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),  # Explicit input shape
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16)
        ])
        
    def perform_feature_selection(self, X, y, k=3):
        """Select most important features using F-regression"""
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_mask = selector.get_support()
        self.selected_features = [f for f, selected in zip(self.features, selected_mask) if selected]
        
        # Rebuild model with correct input dimension if needed
        if len(self.selected_features) != len(self.features):
            self.model = self._build_model(len(self.selected_features))
            
        return X_selected

class BlockchainConnector:
    def __init__(self, infura_url, contract_address, private_key, abi_file):
        # Initialize Web3 connection
        self.web3 = Web3(Web3.HTTPProvider(infura_url))
        assert self.web3.is_connected(), "Failed to connect to Infura"
        
        # Load contract ABI and create contract instance
        with open(abi_file, 'r') as f:
            abi = json.load(f)
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)
        
        # Set up account
        self.private_key = private_key
        self.account = self.web3.eth.account.from_key(private_key).address

def submit_model_update(self, storage_identifier, round_number):
    """Submit model update reference to blockchain"""
    try:
        tx = self.contract.functions.submitModelUpdate(
            str(storage_identifier),  # Make sure it's a string
            int(round_number)         # Make sure it's an integer
        ).build_transaction({
            'from': self.account,
            'nonce': self.web3.eth.get_transaction_count(self.account),
            'gas': 2000000,
            'gasPrice': self.web3.to_wei('20', 'gwei')
        })
        
        signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return tx_hash.hex()
    except Exception as e:
        print(f"Detailed error in submit_model_update: {str(e)}")
        return None

class VFLSystem:
    def __init__(self, mu=0.01, contract_address=None, private_key=None, infura_url=None, abi_file=None):
        self.mu = mu
        self.client1 = Client("Price_Client", ['Price', '1h %', '24h %', '7d %'])
        self.client2 = Client("Market_Client", ['Market Cap', 'Volume (24h)', 'Circulating Supply'])
        
        # Initialize storage system
        self.storage = ModelStorage()
        
        # Initialize blockchain connector if credentials provided
        if all([contract_address, private_key, infura_url, abi_file]):
            self.blockchain_connector = BlockchainConnector(
                infura_url=infura_url,
                contract_address=contract_address,
                private_key=private_key,
                abi_file=abi_file
            )
        else:
            self.blockchain_connector = None
    def evaluate(self, X1_test, X2_test, y_test):
    #Evaluate the model performance on test data"""
    # Convert test data to tensors
    X1_test = tf.convert_to_tensor(X1_test, dtype=tf.float32)
    X2_test = tf.convert_to_tensor(X2_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    # Get predictions from both clients
    pred1 = self.client1.model(X1_test)
    pred2 = self.client2.model(X2_test)
    
    # Combine predictions (average in this case)
    combined_pred = (pred1 + pred2) / 2
    
    # Calculate MSE loss
    loss = tf.keras.losses.MSE(y_test, combined_pred)
    return tf.reduce_mean(loss)
    def preprocess_data(self, data):
        """Prepare and preprocess data for both clients"""
        # Handle missing values and convert to numeric
        for col in self.client1.features + self.client2.features:
            data[col] = pd.to_numeric(data[col].replace(['N/A', ''], np.nan), errors='coerce')
        data = data.fillna(0)
        
        # Split features between clients
        X1 = data[self.client1.features]
        X2 = data[self.client2.features]
        y = data['24h %']
        
        # Feature selection and scaling
        X1_selected = self.client1.perform_feature_selection(X1, y)
        X2_selected = self.client2.perform_feature_selection(X2, y)
        
        scaler = StandardScaler()
        X1_scaled = scaler.fit_transform(X1_selected)
        X2_scaled = scaler.fit_transform(X2_selected)
        
        print(f"Selected features for {self.client1.name}: {self.client1.selected_features}")
        print(f"Selected features for {self.client2.name}: {self.client2.selected_features}")
        
        return X1_scaled, X2_scaled, y

    

    def train(self, data, epochs=10, batch_size=32):
        """Train the federated model and store updates locally"""
        X1, X2, y = self.preprocess_data(data)
        
        X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
            X1, X2, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X1_train = tf.convert_to_tensor(X1_train, dtype=tf.float32)
        X2_train = tf.convert_to_tensor(X2_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(epochs):
            # Training loop implementation remains the same
            # ... [Previous training loop code]
            
            # Store model updates locally and on blockchain
            # Inside the train method, replace the weights dictionary creation with:
            weights = {
                        "client1": [w.tolist() for w in self.client1.model.get_weights()],
                         "client2": [w.tolist() for w in self.client2.model.get_weights()]
}
            # Save to local storage
            storage_id = self.storage.save_model_weights(weights, epoch)
            
            # Submit to blockchain if configured
            if self.blockchain_connector:
                try:
                    tx_hash = self.blockchain_connector.submit_model_update(storage_id, epoch)
                    print(f"Model update for epoch {epoch} submitted. Storage ID: {storage_id}, Tx Hash: {tx_hash}")
                except Exception as e:
                    print(f"Error submitting update to blockchain: {e}")
                    
            # Evaluate periodically
            if (epoch + 1) % 5 == 0:
                test_loss = self.evaluate(X1_test, X2_test, y_test)
                print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
#configration
config = {
    "infura_url": "YOUR_INFURA_URL",
    "contract_address": "YOUR_CONTRACT_ADDRESS",
    "private_key": "YOUR_WALLET_PRIVATE_KEY",
    "abi_file": "PATH_TO_YOUR_ABI.json"
}


    
    # Load data
    filepath = "C:/Users/keerthan/Desktop/VFL BL/crypto_trends_insights_2024.csv"
    data = pd.read_csv(filepath,encoding='latin-1')
    
    # Initialize and train
    vfl_system = VFLSystem(**config)
    vfl_system.train(data, epochs=20)
