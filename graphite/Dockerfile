FROM python:3.9-slim

WORKDIR service

# установим необходимые библиотеки
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app.py .
COPY model.pkl .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
