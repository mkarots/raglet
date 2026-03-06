# raglet Examples

This directory contains example scripts demonstrating how to use raglet.

## Core Examples

### `simple_test.py`
Basic usage example: create, search, and save a raglet.

### `quick_test.py`
Quick test script for basic functionality.

### `add_files_example.py`
Demonstrates adding files incrementally to an existing raglet.

### `agentic_loop_example.py`
Simple agentic loop example showing how to use raglet with an external LLM (Claude).

## Advanced Examples

### `chat_cli_example.py`
**Interactive agentic chat loop (CLI-style)**

A polished CLI-style example demonstrating how to use raglet in an agentic loop pattern:
1. Search raglet for relevant context
2. Use external LLM (Claude) to generate response
3. Store conversation in raglet for future retrieval

**Usage:**
```bash
# Install optional dependencies
pip install raglet[chat]

# With environment variable (works with directory, zip, or sqlite)
export ANTHROPIC_API_KEY="your-api-key"
python examples/chat_cli_example.py --raglet .raglet/
python examples/chat_cli_example.py --raglet my-raglet.zip
python examples/chat_cli_example.py --raglet knowledge.sqlite

# With API key flag
python examples/chat_cli_example.py --raglet .raglet/ --api-key sk-...

# Docker example
docker run -it -v $(pwd):/data -e ANTHROPIC_API_KEY=sk-... raglet python examples/chat_cli_example.py --raglet /data/.raglet/
```

**Note:** This is a demonstration script showing the "bring your own LLM" pattern. raglet itself is retrieval-only - this script shows one integration example.

## Docker Examples

### `docker-chat-example.sh`
Shell script demonstrating how to run the chat example in Docker.

**Usage:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
./examples/docker-chat-example.sh
```

## Docker Usage

The raglet Docker image includes all core CLI commands:

```bash
# Build Docker image
docker build -t raglet:latest .

# Build raglet from files (creates directory)
docker run -v $(pwd):/data raglet build /data/docs --out /data/.raglet/

# Query raglet (works with any format)
docker run -v $(pwd):/data raglet query --raglet /data/.raglet/ --q "python" --top-k 5
docker run -v $(pwd):/data raglet query --raglet /data/export.zip --q "python" --top-k 5
docker run -v $(pwd):/data raglet query --raglet /data/knowledge.sqlite --q "python" --top-k 5

# Add files incrementally (works with any format)
docker run -v $(pwd):/data raglet add --raglet /data/.raglet/ /data/new-file.txt
docker run -v $(pwd):/data raglet add --raglet /data/knowledge.sqlite /data/new-file.txt

# Package (convert between formats)
docker run -v $(pwd):/data raglet package --raglet /data/.raglet/ --format zip --out /data/export.zip
docker run -v $(pwd):/data raglet package --raglet /data/export.zip --format sqlite --out /data/knowledge.sqlite
docker run -v $(pwd):/data raglet package --raglet /data/knowledge.sqlite --format dir --out /data/.raglet/
```

For chat functionality, install the optional dependencies:
```bash
docker run -it -v $(pwd):/data -e ANTHROPIC_API_KEY=sk-... raglet bash -c "pip install anthropic && python examples/chat_cli_example.py --raglet /data/.raglet/"
```
