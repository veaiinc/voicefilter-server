FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server_cloud.py .

EXPOSE 7860

CMD ["python", "server_cloud.py"]
