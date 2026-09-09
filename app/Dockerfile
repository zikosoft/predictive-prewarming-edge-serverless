FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.30.6 --root-user-action=ignore
COPY telemetry_app.py .
ENV INSTANCE_ID=unset
EXPOSE 8000
CMD ["uvicorn", "telemetry_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
