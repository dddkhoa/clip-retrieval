FROM node:18 as node-base

COPY front /app/front

WORKDIR /app/front
COPY front/package.json .
COPY front/webpack.config.js .

RUN npm install
RUN npm run build

FROM python:3.10-slim as python-base

WORKDIR /app
COPY . /app
COPY --from=node-base /app/front/node_modules /app/front/node_modules
COPY --from=node-base /app/front/build /app/front/build
RUN pip install -e .

EXPOSE 1234

CMD ["sh", "start.sh"]