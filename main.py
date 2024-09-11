from fastapi import FastAPI, Request
import pickle
import uvicorn

# загрузите модель из файла выше
with open("model.pkl", "rb") as f:
    uplift_model = pickle.load(f) 

# создаём приложение FastAPI
app = FastAPI(title="uplift")

@app.post("/predict")
async def predict(request: Request):

	# все данные передаются в json
    data = await request.json()

	# признаки лежат в features, в массиве
    # извлекаем и преобразуем признаки
    features = data["features"]

    # получаем предсказания
    prediction = uplift_model.predict(features)

    return {"predict": prediction[:,1].tolist()}

if __name__ == '__main__':
	# запустите сервис на хосте 0.0.0.0 и порту 5000
    uvicorn_kwargs = dict(
        host='0.0.0.0',
        port=5000,
        log_level="debug",
    )
    uvicorn.run(app, **uvicorn_kwargs)
