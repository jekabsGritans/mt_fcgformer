FROM pytorch/pytorch:latest

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# COPY . . # maybe need this when deploying to cloud and not mapping volumes

CMD ["python", "main.py"]