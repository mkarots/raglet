# Why raglet?

## The Problem

There's a class of knowledge that's small but too big for a prompt.

A codebase. A Slack conversation. A WhatsApp chat export. A folder of meeting notes.
These are small — maybe a few megabytes — but they don't fit in a context window.
And they definitely don't justify a vector database, a server, or any infrastructure.

Today, your options are:
1. **Copy-paste into an LLM** — works until it doesn't fit.
2. **Set up a RAG pipeline** — embeddings, vector DB, retrieval, chunking, config. Way too much for a small problem.
3. **Just... remember** — scroll through Slack, grep the codebase, hope you find it.

There's no lightweight, portable way to make a small pile of text searchable and LLM-ready.

## The Idea

raglet is portable memory.

It takes small context — source files, conversations, notes — and turns it into a
single `.raglet` file that you can save, share, commit, or carry around. Load it
anywhere, search it instantly, and get retrieval-ready context for whatever LLM
or tool you're already using.

No server. No API keys. No infrastructure. Just a Python object and a file.

## Principles

- **Portable.** One file. Save it, git commit it, email it, drag it to another machine.
- **Small by design.** Built for workspace-scale problems: codebases, conversations, notes. Not the internet.
- **Retrieval only.** raglet finds the right chunks. You decide what to do with them. Bring your own LLM.
- **Open format.** The `.raglet` file is easily decodable. Embeddings are extractable. No lock-in.
- **Zero infrastructure.** `pip install raglet`. That's it.

## Who Is This For?

- Developers building LLM-powered tools who need a retrieval component that just works
- Anyone who wants to make a small corpus searchable without setting up infrastructure
- Projects where the context is ephemeral — it matters now, not forever

## What This Is NOT

- Not a vector database
- Not a full RAG pipeline (no LLM, no generation)
- Not built for large-scale datasets
- Not a web app or a service

## The Bet

Most RAG problems are small. A few files. A few thousand chunks. If you make
retrieval as simple as `pip install` and a `.raglet` file, it becomes a building
block that shows up everywhere — in scripts, agents, CLI tools, notebooks — anywhere
someone needs to give an LLM a small, searchable memory.