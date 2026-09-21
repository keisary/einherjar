# EINHERJAR — image portable (API FastAPI + boucle d'inference dans un seul process).
#
# Un seul process est une contrainte du systeme : le reactor Twisted de cTrader est un
# singleton, deux adaptateurs dans le meme process se font tomber l'un l'autre.
#
# Construction :  docker build -t einherjar .
# Execution   :  docker run --env-file .env -p 8000:8000 -v "$PWD/data:/app/data" einherjar

FROM python:3.11-slim

# PYTHONUNBUFFERED : les logs sortent immediatement (docker logs, journalctl).
# NUMBA_CACHE_DIR  : le cache de compilation JIT part dans le volume `data/`, donc il
#                    survit aux redemarrages — sans lui, chaque demarrage recompile
#                    tout Numba (mesure : 93 s sur un poste multicœur, 30-50 min sur
#                    0,1 CPU). C'est le principal gain d'une VM ou d'un conteneur
#                    avec volume par rapport a un hebergement ephemere.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/app/data/numba_cache \
    PORT=8000

# libgomp1 est requis par Numba (parallelisme), curl sert au HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Le dossier d'etat doit exister meme sans volume monte : DuckDB et le cache Numba y
# ecrivent (un `docker run` sans -v ne doit pas echouer).
RUN mkdir -p /app/data

# 1) dependances d'execution, versions epinglees et verifiees en exploitation
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) client cTrader SANS ses dependances : il epingle protobuf==3.20.1, dont il
#    n'existe pas de roue pour Python 3.11, alors que ses modules generes
#    fonctionnent avec le protobuf moderne installe ci-dessus.
RUN pip install --no-cache-dir --no-deps ctrader-open-api==0.9.2

# 3) le projet lui-meme (le paquet declare ses dependances : deja installees)
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Etat runtime : base DuckDB (journal, fills, courbe d'equity), parquets OHLCV live,
# cache Numba. A monter en volume pour survivre aux redemarrages.
VOLUME ["/app/data"]

EXPOSE 8000

# Le demarrage comprend l'amorcage broker puis l'echauffement JIT : on laisse 180 s
# avant de considerer l'echec comme definitif.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

CMD ["python", "main.py"]
