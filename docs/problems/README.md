# Problems

This directory contains problem statements and context for raglet.

## Files

- [00-problem-statement.md](00-problem-statement.md) - Core problem statement (from WHY.md)

## Problem Summary

There's a class of knowledge that's **small but too big for a prompt**:
- A codebase
- A Slack conversation
- A WhatsApp chat export
- A folder of meeting notes

These are small (a few megabytes) but don't fit in a context window. They also don't justify a vector database, server, or infrastructure setup.

**Current options are inadequate:**
1. Copy-paste into LLM → works until it doesn't fit
2. Set up RAG pipeline → way too much infrastructure
3. Just remember → scroll, grep, hope you find it

**Gap:** No lightweight, portable way to make a small pile of text searchable and LLM-ready.
