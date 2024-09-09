import pandas as pd
import numpy as np

from sklearn.metrics import auc
import matplotlib.pyplot as plt


# Шаг 1: Сбор данных

# генерация данных для 100 пользователей
np.random.seed(42)
n = 100
treatment = np.array([1]*50 + [0]*50)
outcome = np.concatenate([np.random.choice([1, 0], p=[0.2, 0.8], size=50),
                          np.random.choice([1, 0], p=[0.1, 0.9], size=50)])
uplift_prediction = np.random.rand(n)

# создание DataFrame
data = {
    'user_id': range(1, n+1),
    'treatment': treatment,
    'outcome': outcome,
    'uplift_prediction': uplift_prediction
}

df = pd.DataFrame(data)
print(df.head())

# Шаг 2: Сортировка данных по uplift_prediction

# сортировка данных по uplift_prediction
df = df.sort_values(by='uplift_prediction', ascending=False).reset_index(drop=True)
print(df.head())

# Шаг 3: Инициализация переменных для расчёта CGain и Random

# инициализация переменных

nt = 0
nt_1 = 0
nc = 0
nc_1 = 0
cgain = []
random = []
optimum = []
incremental_purchases = 0

opt_incremental_purchases = (df["treatment"] * df["outcome"]).sum()
negative_incremental_purchases = ((1 - df["treatment"]) * df["outcome"]).sum()

# Шаг 4: Расчёт CGain, Random и Optimum

# расчёт CGain, Random и Optimum

for i, row in df.iterrows():
    if row['outcome'] == 1:
        if row['treatment'] == 1:
            incremental_purchases += 1
        else:
            incremental_purchases -= 1

    cgain.append(incremental_purchases)

    random.append((opt_incremental_purchases - negative_incremental_purchases) * i / len(df))
    optimum.append(min(opt_incremental_purchases, i) + min(0, len(df) - i - negative_incremental_purchases - 1))

# Шаг 5: Расчёт метрик

# расчёт площади под кривыми
qini_auc = auc(range(1, len(cgain) + 1), cgain)
random_auc = auc(range(1, len(random) + 1), random)

# расчёт Qini Score
qini_score = qini_auc - random_auc

# вывод Qini Score
print(f'Qini Score: {qini_score}')

# вывод Optimum Qini Score
print(f'Optimum Qini Score: {auc(range(1, len(optimum) + 1), optimum) - random_auc}') 

# Шаг 6: Построение графика

# построение графиков
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cgain) + 1), cgain, label='Model', color='blue')
plt.plot(range(1, len(random) + 1), random, label='Random', color='red', linestyle='--')
plt.plot(range(1, len(optimum) + 1), optimum, label='Optimum', color='green', linestyle='--')
plt.xlabel('Number of users targeted')
plt.ylabel('Number of Incremental Purchases')
plt.title('Gains Chart for Uplift (Qini Curve) with Random')
plt.legend()
plt.grid(True)
plt.show()

