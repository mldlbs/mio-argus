FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    scrot \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyautogui \
    Pillow \
    python-xlib

WORKDIR /app

COPY collect_vm.py .

CMD ["python", "/app/collect_vm.py"]