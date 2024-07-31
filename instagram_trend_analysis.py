import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Try different encodings if UTF-8 fails
encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'windows-1252']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv('instagram_data.csv', encoding=encoding)
        print(f"Successfully read the file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the file with encoding: {encoding}")

if df is None:
    raise ValueError("Failed to read the CSV file with all attempted encodings.")

# Preprocess the data
df = df.fillna(0)  # Fill missing values with 0

# Handle categorical data
# Example: Encoding categorical columns if they exist
from sklearn.preprocessing import LabelEncoder

if 'Caption' in df.columns:
    le_caption = LabelEncoder()
    df['Caption'] = le_caption.fit_transform(df['Caption'])
    
if 'Hashtags' in df.columns:
    le_hashtags = LabelEncoder()
    df['Hashtags'] = le_hashtags.fit_transform(df['Hashtags'])

# Ensure 'Likes' column exists and is numeric
if 'Likes' not in df.columns:
    raise ValueError("The 'Likes' column is missing from the dataset")

# Splitting the data into features and target variable
X = df.drop(['Likes'], axis=1)
y = df['Likes']

# Convert features to numeric if they are not
X = pd.get_dummies(X)  # Convert categorical features to dummy variables if needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualize the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Likes')
plt.ylabel('Predicted Likes')
plt.title('Actual vs Predicted Likes')
plt.show()
