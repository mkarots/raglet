#!/bin/bash
# Example: Using raglet chat in Docker
#
# This demonstrates how to run the chat example script in a Docker container.
# The raglet Docker image includes all core functionality (build, query, add, package).
# For chat functionality, you need to install the optional 'chat' dependencies.
#
# Usage:
#   ./examples/docker-chat-example.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}raglet Docker Chat Example${NC}"
echo ""

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable is not set."
    echo "Set it with: export ANTHROPIC_API_KEY='your-api-key'"
    exit 1
fi

# Build Docker image (if not already built)
echo -e "${GREEN}Building raglet Docker image...${NC}"
docker build -t raglet:latest .

# Create a volume mount point for raglet files
RAGLET_DIR=$(pwd)/docker-raglet-data
mkdir -p "$RAGLET_DIR"

echo -e "${GREEN}Running chat example in Docker...${NC}"
echo ""
echo "Note: The chat example requires the 'chat' optional dependencies."
echo "Install them in the container or use a custom Dockerfile."
echo ""

# Run chat example in Docker
# Mount current directory to access examples/
# Mount raglet data directory for persistence
docker run -it \
    -v "$(pwd):/app" \
    -v "$RAGLET_DIR:/data" \
    -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    --entrypoint /bin/bash \
    raglet:latest \
    -c "pip install anthropic && python /app/examples/chat_cli_example.py --raglet /data/.raglet/"

echo ""
echo -e "${GREEN}Done! Raglet files are saved in: $RAGLET_DIR${NC}"
