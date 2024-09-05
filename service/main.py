from fastapi import FastAPI, Request
import pickle
import numpy as np
import uvicorn

# загрузите модель из файла выше
# ваш код здесь
with open("model.pkl", "rb") as f:
    uplift_model = pickle.load(f) 

# создаём приложение FastAPI
app = FastAPI(title="uplift")


@app.post("/predict")
async def predict(request: Request):
	# признаки лежат в features, в массиве
    try:
        data = await request.json()
        # извлекаем и преобразуем признаки
        # ваш код здесь
        features = data.get('features')
    except Exception as error:
        print(error)
        return {"predict": 'error'}
        
    # получаем предсказания
    # ваш код здесь
    try:
        features = features if np.array(features).ndim == 2 else [features]
        prediction = uplift_model.predict(features)[:,1] 
        return {"predict": prediction.tolist()}
    except Exception as error:
        print(error)
        return {"predict": 'error'}

if __name__ == '__main__':
    # запустите сервис на хосте 0.0.0.0 и порту 5000
    uvicorn.run(
        "main:app", host="0.0.0.0", port=5000, reload=True
    )
