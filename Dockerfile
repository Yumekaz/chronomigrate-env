FROM ghcr.io/meta-pytorch/openenv-base:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    postgresql-17 \
    postgresql-client-17 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user

ENV PGDATA=/home/user/pgdata
ENV PGPORT=5433
ENV ENABLE_WEB_INTERFACE=true

RUN chmod +x /home/user/app/scripts/init_db.sh

EXPOSE 7860

CMD ["/home/user/app/scripts/init_db.sh"]
