# подготовка данных и обучение модели
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# пример данных
data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5, 6],
    'treatment': [1, 0, 1, 0, 1, 0],
    'num_trips': [10, 8, 12, 7, 9, 6],
    'avg_trip_cost': [15, 12, 14, 11, 13, 10],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'location': ['City A', 'City B', 'City A', 'City B', 'City A', 'City B'],
    'target': [5, 2, 7, 1, 4, 0]
})

# трансформация целевой переменной
data['target_transformed'] = data.apply(
    lambda row: row['target'] if row['treatment'] == 1 else -row['target'], axis=1)

# целевая переменная и признаки
y = data['target_transformed']
X = data.drop(['target', 'target_transformed', 'user_id'], axis=1)

# разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# предобработка данных
numeric_features = ['num_trips', 'avg_trip_cost']
categorical_features = ['gender', 'location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# модель
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# обучение модели
model.fit(X_train, y_train)

# оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Predictions:", y_pred)
