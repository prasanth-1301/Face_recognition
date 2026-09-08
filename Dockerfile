FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    cython \
    numpy==1.26.4

RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    pandas==2.2.3 \
    opencv-python-headless==4.10.0.84 \
    onnxruntime==1.19.2

RUN pip install --no-cache-dir insightface==0.7.3

COPY . .

EXPOSE 7860

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

RUN pip install --no-cache-dir insightface==0.7.3