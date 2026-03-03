# Decision Log

This directory contains architectural and design decisions for raglet.

## Decisions

- [001: Library, Not Service](001-library-not-service.md) - Python library approach
- [002: SOLID Architecture](002-solid-architecture.md) - SOLID principles and structure
- [003: Shallow Interface, Deep Configuration](003-shallow-interface-deep-config.md) - API vs config philosophy
- [004: Nested Configuration](004-nested-configuration.md) - Configuration structure
- [005: Portable File Format](005-portable-file-format.md) - .raglet file format
- [006: Technology Stack](006-technology-stack.md) - Dependencies and tools
- [007: Agent Tools](007-agent-tools.md) - Anthropic tool integration

## Format

Each decision follows this structure:
- **Date:** When decision was made
- **Status:** Accepted/Rejected/Proposed
- **Context:** What problem this solves
- **Decision:** What we decided
- **Rationale:** Why we decided this
- **Consequences:** What this means
- **Alternatives:** What we considered

## Adding New Decisions

When making significant architectural or design decisions:
1. Create new file: `XXX-decision-name.md`
2. Follow the format above
3. Update this README
4. Reference in relevant documentation
