FROM rust

COPY . .

RUN ["cargo", "test"]

