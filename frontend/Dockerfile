FROM node:20-alpine
ENV NODE_ENV=development

WORKDIR /frontend

COPY package.json package-lock.json  
RUN npm install

COPY . .

RUN npm run dev