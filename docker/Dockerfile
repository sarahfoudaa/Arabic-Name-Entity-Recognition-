FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./nerModel ./nerModel
COPY ./tokenizer ./tokenizer

COPY ./app.py ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
