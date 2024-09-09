import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from causalml.inference.tree import UpliftTreeClassifier


# Загружаем данные
df = pd.read_csv("discountuplift.csv", sep="\t")
print(f"df: {df.shape}")
print(f"{df.head(1)}")

# добавим признак `old target` — принимает значение 1, если была покупка, и 0 в противном случае
df['old_target'] = df['target_class'] % 2

# выделим колонки с факторами и колонку с фактом «целевого воздействия» — выдачи скидки
feature_cols = ['recency', 'history', 'used_discount', 'used_bogo', 'is_referral',
                'zip_code_Rural', 'zip_code_Surburban', 'zip_code_Urban',
                'channel_Multichannel', 'channel_Phone', 'channel_Web']
target_col = 'old_target'
treatment_col = 'treatment' 

# разобъём нашу выборку на тестовую и валидационную
df_train, df_test = train_test_split(
    df,
    stratify=df[[treatment_col, target_col]],
    random_state=1,
    test_size=0.25,
)

# создадим базовое uplift-дерево
uplift_model = UpliftTreeClassifier(
    max_depth=5,
    min_samples_leaf=200,
    min_samples_treatment=50,
    n_reg=100,
    evaluationFunction='ED',
    control_name='0',
)
print(uplift_model)

# обучение модели
uplift_model.fit(
    X=df_train[feature_cols].values,
    treatment=df_train[treatment_col].apply(str).values,
    y=df_train[target_col].values
)

# Сохраняем модель
with open("model.pkl", "wb") as f:
    pickle.dump(uplift_model, f)
print("model saved")
