# Utiliser l'image Python officielle
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système (pour compiler certains paquets Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances
COPY requirements.txt .
# Installer la version CPU de PyTorch en premier pour éviter le téléchargement de 4 Go de librairies NVIDIA (qui font crasher Docker Desktop)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copier tout le code source du backend
COPY . .

# Exposer le port de l'application
EXPOSE 8000

# Commande pour démarrer FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
