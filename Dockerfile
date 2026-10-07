FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY results/ results/

ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "skillrouterbench.classifiers.serve:app", "--host", "0.0.0.0", "--port", "8000"]
