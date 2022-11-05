FROM python:3.8

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code


COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000