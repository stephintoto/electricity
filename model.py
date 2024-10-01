import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv"
df = pd.read_csv(url)

df = df.dropna()  
X = df[['Consumption']]  
y = df['Price']  

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'electricity_price_model.pkl')
