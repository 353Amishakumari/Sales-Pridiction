FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

CMD ["python", "app.py"]