FROM ollama/ollama:latest

COPY ./ollama_init.sh /tmp/run-ollama.sh


WORKDIR /tmp

RUN chmod +x run-ollama.sh \
    && ./run-ollama.sh

EXPOSE 11434

ENTRYPOINT ["ollama", "serve"]