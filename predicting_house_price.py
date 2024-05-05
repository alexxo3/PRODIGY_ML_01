import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
train_data = pd.read_csv('/data/train.csv')
test_data = pd.read_csv('/data/test.csv')

# Extract features and target variable
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']].values
y_train = train_data['SalePrice'].values
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']].values

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_predictions = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
print("Train RMSE:", train_rmse)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: GrLivArea vs. SalePrice
axs[0].scatter(train_data['GrLivArea'], train_data['SalePrice'], alpha=0.5)
axs[0].set_xlabel('GrLivArea')
axs[0].set_ylabel('SalePrice')
axs[0].set_title('GrLivArea vs. SalePrice')

# Plot 2: BedroomAbvGr vs. SalePrice
axs[1].scatter(train_data['BedroomAbvGr'], train_data['SalePrice'], alpha=0.5)
axs[1].set_xlabel('BedroomAbvGr')
axs[1].set_ylabel('SalePrice')
axs[1].set_title('BedroomAbvGr vs. SalePrice')

# Plot 3: FullBath vs. SalePrice
axs[2].scatter(train_data['FullBath'], train_data['SalePrice'], alpha=0.5)
axs[2].set_xlabel('FullBath')
axs[2].set_ylabel('SalePrice')
axs[2].set_title('FullBath vs. SalePrice')

plt.tight_layout()
plt.show()

# Make predictions on test data
test_predictions = model.predict(X_test)

# Output predictions
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})
submission.to_csv('/data/sample_submission.csv', index=False)
