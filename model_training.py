import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# List of file paths
file_paths = [
    r'C:\Users\agamb\Downloads\E-commerce.csv',
    r'C:\Users\agamb\Downloads\sldp datasets IV\E-commerce Customer Behavior - Sheet1.csv',
    r'C:\Users\agamb\Downloads\sldp datasets IV\E-commerce (1).csv',
    r'C:\Users\agamb\Downloads\sldp datasets IV\ecommerce_customer_data.csv',
]

# Columns to use
columns_to_use = ['Age', 'Gender']

dataframes = []
for file in file_paths:
    try:
        df = pd.read_csv(file)
        if set(columns_to_use).issubset(df.columns):  # Check if required columns exist
            df = df[columns_to_use].dropna()
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            dataframes.append(df)
        else:
            print(f"Skipping {file} due to missing columns.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Concatenate all dataframes
if dataframes:
    final_data = pd.concat(dataframes, ignore_index=True)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(final_data)
    
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    # Train-test split
    X_train, X_test = train_test_split(data_tensor, test_size=0.2, random_state=42)

    # Define Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=2):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # Initialize model
    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(input_dim)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    # Train autoencoder
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = autoencoder(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    # Extract reduced dimensions
    encoded_data = autoencoder.encoder(data_tensor).detach().numpy()
    encoded_df = pd.DataFrame(encoded_data, columns=['Encoded Feature 1', 'Encoded Feature 2'])

    # Visualize
    plt.scatter(encoded_df['Encoded Feature 1'], encoded_df['Encoded Feature 2'], alpha=0.5)
    plt.title('Autoencoder Reduced Data')
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')
    plt.show()
else:
    print("No valid data to process.")
