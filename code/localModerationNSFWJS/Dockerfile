FROM node:20-bullseye-slim

WORKDIR /app

# Build deps for tfjs-node fallback builds (most platforms use prebuilt binaries)
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 make g++ \
  && rm -rf /var/lib/apt/lists/*

COPY package.json ./
RUN npm install --omit=dev

COPY app.js ./

ENV PORT=5082
EXPOSE 5082

CMD ["node", "app.js"]
