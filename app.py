import pickle
import uvicorn
from time import perf_counter

from fastapi import FastAPI, Request, HTTPException
from statsd import StatsClient


# загрузите модель из файла выше
with open("model.pkl", "rb") as f:
    uplift_model = pickle.load(f) 

# создаём приложение FastAPI
app = FastAPI(title="uplift")

stats_client = StatsClient(host="localhost", port=8125, prefix="uplift")


@app.post("/predict")
async def predict(request: Request):

    # запомним время начала обработки запроса
    start_time = perf_counter()

	# все данные передаются в json
    try:
        data = await request.json()
    except Exception as e:
        stats_client.incr("errors.invalid_json")
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    # извлекаем и преобразуем признаки
    try:
        features = data["features"]
    except Exception as e:
        stats_client.incr("errors.invalid_features")
        raise HTTPException(status_code=400, detail="Invalid features format")

    # получаем предсказания
    try:
        prediction = uplift_model.predict(features)[:,1]
    except Exception as e:
        stats_client.incr("errors.model_prediction")
        raise HTTPException(status_code=500, detail="Model prediction error")

    # посчитаем время обработки запроса в секундах как разницу 
    # между текущим временем и start_time
    response_time = perf_counter() - start_time

    stats_client.timing("response_time", response_time)
    stats_client.incr("response_code.200")
    stats_client.gauge("predictions", prediction[0])

    return {"predict": prediction.tolist()}

if __name__ == '__main__':
	
    # запустите сервис на хосте 0.0.0.0 и порту 5000
    uvicorn_kwargs = dict(
        host='0.0.0.0',
        port=5000,
        log_level="debug",
    )
    uvicorn.run(app, **uvicorn_kwargs)
