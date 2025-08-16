FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libgtk-3-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    libopenblas-dev liblapack-dev \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# (Опционально) Сборка OpenFace 2.0 внутри образа
# Если сборка займёт слишком много времени на вашей машине, можно:
# 1) собрать заранее и примонтировать бинарь в /opt/OpenFace/build/bin/FeatureExtraction
# 2) или указать путь к бинарю через переменную окружения OPENFACE_CMD
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git /opt/OpenFace \
 && mkdir -p /opt/OpenFace/build \
 && cd /opt/OpenFace/build \
 && cmake .. \
 && make -j"$(nproc)"

COPY . /app

ENV OPENFACE_CMD=/opt/OpenFace/build/bin/FeatureExtraction

EXPOSE 8501
