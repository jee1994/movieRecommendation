FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/models

EXPOSE 50052

RUN echo '#!/bin/bash\npython -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. embedding_service.proto\npython embedding_server.py "$@"' > /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh", "--model-path=/app/models/user_model_final.pt", "--port=50052"]