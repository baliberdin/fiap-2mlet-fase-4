FROM node:22.12.0-alpine

WORKDIR /opt/book-store

COPY ./bin ./bin
COPY ./public ./public
COPY ./routes ./routes
COPY ./views ./views
COPY app.js .
COPY package.json .
COPY package-lock.json .

RUN npm install

CMD ["npm", "start"]