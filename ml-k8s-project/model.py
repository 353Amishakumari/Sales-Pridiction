import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample data
data = {
    "experience": [1, 2, 3, 4, 5],
    "marketing": [10000, 20000, 30000, 40000, 50000],
    "store_size": [100, 200, 300, 400, 500],
    "sales": [200, 300, 500, 700, 900]
}

df = pd.DataFrame(data)

X = df[["experience", "marketing", "store_size"]]  # 3 features
y = df["sales"]

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained with 3 features ✅")