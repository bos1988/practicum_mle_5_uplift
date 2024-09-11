FROM python:3.9-slim

WORKDIR /uplift

# установим необходимые библиотеки
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY model.pkl .
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
