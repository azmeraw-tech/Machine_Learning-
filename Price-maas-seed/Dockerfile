FROM python:3.8.8-slim
WORKDIR /app

# install the bits we need
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
    curl jq unzip \
&& pip install --upgrade pip awscli \
&& rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
