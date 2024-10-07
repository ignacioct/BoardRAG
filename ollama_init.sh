echo "Starting Ollama server..."
ollama serve&


echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

ollama list
ollama pull nomic-embed-text

ollama serve&
ollama list
ollama pull mistral