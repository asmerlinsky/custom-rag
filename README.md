# Custom RAG

### Installing

- Install Ollama and desired model
   ```sh
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull phi
   ```
- Install module
   ```sh
   poetry install
   ```
- Place documents in `./data` dir and add to database
   ```sh
   poetry run python main.py load_documents
   ```
- Ask questions
   ```sh
    poetry run python main.py ask "what can you tell me about something?"
   ```
