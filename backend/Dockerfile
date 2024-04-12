FROM python:3.10.10

WORKDIR /backend
RUN apt update

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]