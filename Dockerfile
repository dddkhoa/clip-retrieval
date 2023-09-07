FROM python:3.10-slim as python-base

WORKDIR /app
COPY . /app
RUN pip install -e .

EXPOSE 8080

FROM node:16-alpine
WORKDIR /app/front
COPY front/package.json /app/front
RUN npm install
RUN npm run build

CMD ["sh", "./start.sh"]