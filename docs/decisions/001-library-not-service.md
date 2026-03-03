# Decision 001: Library, Not Service

**Date:** December 2024  
**Status:** Accepted

## Context

Initial planning documents included web service approaches (FastAPI, React frontend, API endpoints).

## Decision

Build raglet as a **Python library**, not a web service or API.

## Rationale

- Aligns with "zero infrastructure" principle from WHY.md
- Portable `.raglet` files don't need servers
- Easier to integrate into existing workflows
- Can be used directly by agents
- Simpler deployment (`pip install`)

## Consequences

- No FastAPI backend
- No REST API endpoints
- No React frontend
- Library-first approach
- Focus on Python API

## Alternatives Considered

- FastAPI web service (rejected - requires infrastructure)
- Next.js web app (rejected - doesn't match portable file vision)
