FROM docker.1ms.run/library/python:3.12-slim

ENV HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" \
    NO_PROXY="*" no_proxy="*"

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    pillow \
    numpy \
    transformers \
    fastapi \
    uvicorn \
    pydantic \
    pydantic-settings \
    python-multipart

WORKDIR /app

# Copy pre-downloaded HuggingFace models
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_CACHE=/app/hf_cache
COPY hf_cache/ /app/hf_cache/

COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
