FROM node:22-slim

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
      python3=3.11.2-1+b1 \
      python-is-python3=3.11.2-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json ./
COPY packages/shared/package.json ./packages/shared/package.json
COPY apps/web/package.json ./apps/web/package.json
RUN --mount=type=cache,target=/root/.npm,sharing=locked npm ci
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    OXIDE_VERSION="$(node -p "require('/app/apps/web/node_modules/@tailwindcss/oxide/package.json').version")" \
    && NEXT_VERSION="$(node -p "require('next/package.json').version")" \
    && npm install --workspace web --no-save \
      "@tailwindcss/oxide-linux-x64-gnu@${OXIDE_VERSION}" \
      "@next/swc-linux-x64-gnu@${NEXT_VERSION}"

COPY packages/shared ./packages/shared
COPY apps/web ./apps/web

WORKDIR /app/apps/web

ENV NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]
