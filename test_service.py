import requests
import pandas as pd

# загружаем DataFrame
discount = pd.read_csv("discountuplift.csv", sep="\t")

feature_cols = [
    "recency",
    "history",
    "used_discount",
    "used_bogo",
    "is_referral",
    "zip_code_Rural",
    "zip_code_Surburban",
    "zip_code_Urban",
    "channel_Multichannel",
    "channel_Phone",
    "channel_Web",
]

# пример данных для запроса
data = {
    "features": discount[feature_cols].sample(3).values.tolist()
}
print('Запрос среверу:', data)

# URL сервиса
url = "http://localhost:5000/predict"

# выполнение POST-запроса
response = requests.post(url, json=data)

# проверка ответа
if response.status_code == 200:
    print("Ответ сервера:", response.json())
else:
    print("Ошибка:", response.status_code, response.text)
