# SuryaIQ — reproducible container. Multi-service launched via docker-compose.
FROM python:3.11-slim

WORKDIR /app

# System deps: libgomp for scikit-learn / shap, build tools for scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the panel + paper analysis + SHAP so the container is ready to serve
RUN cd src && python build_panel.py && python paper_analysis.py && python explain.py

EXPOSE 8000 8530

CMD ["bash", "-lc", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
     streamlit run app/streamlit_app.py --server.port 8530 --server.address 0.0.0.0 --server.headless true"]
