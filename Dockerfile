FROM rust

COPY . .

CMD ["cargo", "test"]

