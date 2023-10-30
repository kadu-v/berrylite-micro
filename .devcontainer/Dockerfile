FROM rust:latest

RUN apt-get update && apt-get install -y gnuplot
RUN rustup install nightly 
RUN rustup default nightly
RUN rustup component add clippy rls rust-analysis rust-src rust-docs rustfmt rust-analyzer

RUN apt-get update && apt-get install python3-pip
RUN pip3 install tensorflow --break-system-packages
