FROM python:3.10.16

WORKDIR /opt/stock-predictions-api

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./main.py ./main.py
COPY ./api ./api
COPY ./jobs ./jobs
COPY ./configuration ./configuration
COPY ./database ./database
COPY ./services ./services
COPY ./secrets ./secrets
COPY ./env.yaml ./env.yaml
COPY ./logging.conf ./logging.conf

CMD ["fastapi", "run", "main.py", "--proxy-headers", "--port", "80"]