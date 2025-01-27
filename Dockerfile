FROM ubuntu:latest
LABEL authors="Daniel"

ENTRYPOINT ["top", "-b"]