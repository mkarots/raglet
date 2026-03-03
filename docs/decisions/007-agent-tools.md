# Decision 007: Agent Tools Integration

**Date:** December 2024  
**Status:** Accepted

## Context

Agents need programmatic access to raglet functionality.

## Decision

Expose as Anthropic tools: `create_raglet`, `search_raglet`, `get_raglet_info`.

## Rationale

- Agents need programmatic access
- Maintains "zero infrastructure" (library, not API)
- Portable files work well for agent workflows
- Can be adapted to other frameworks

## Consequences

- Three tool functions
- Tool schemas (JSON)
- Zero infrastructure (agents use library directly)
- Portable files enable agent workflows

## Tools

1. `create_raglet` - Create .raglet file from documents
2. `search_raglet` - Search a .raglet file
3. `get_raglet_info` - Get metadata about file
