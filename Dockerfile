FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y tesseract-ocr tesseract-ocr-por \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .

COPY .pip/pip.conf /root/.pip/pip.conf

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src

EXPOSE 50051

CMD ["python", "-m", "ocr_pypi.server"]
