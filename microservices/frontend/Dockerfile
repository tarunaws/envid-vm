FROM node:18-alpine AS build

WORKDIR /app

COPY microservices/frontend/code/app/package.json microservices/frontend/code/app/package-lock.json* ./
RUN npm install

COPY microservices/frontend/code/app/ ./
RUN npm run build

FROM node:18-alpine

WORKDIR /app
COPY microservices/frontend/code/app/package.json microservices/frontend/code/app/package-lock.json* ./
RUN npm install --omit=dev

COPY --from=build /app/build /app/build

EXPOSE 3000

CMD ["node_modules/.bin/serve", "-s", "build", "-l", "3000"]
