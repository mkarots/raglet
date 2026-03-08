# Proposals

This directory contains design proposals and explorations.

## Files

- [DEEP_CONFIG_PROPOSAL.md](DEEP_CONFIG_PROPOSAL.md) - Deep configuration system proposal
- [CONFIG_EXPLORATION.md](CONFIG_EXPLORATION.md) - Configuration pattern exploration
- [PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md) - Performance bottleneck analysis and fixes
- [CHUNK_SIZE_TOKEN_OVERFLOW.md](CHUNK_SIZE_TOKEN_OVERFLOW.md) - Silent token overflow in chunk sizing
- [EMBEDDING_THROUGHPUT_INVESTIGATION.md](EMBEDDING_THROUGHPUT_INVESTIGATION.md) - Embedding throughput profiling and optimisation results

## Status

- **DEEP_CONFIG_PROPOSAL.md** - ✅ Accepted (see decisions/003, 004)
- **CONFIG_EXPLORATION.md** - ✅ Completed (led to deep config proposal)
- **PERFORMANCE_IMPROVEMENTS.md** - 🔄 In progress
- **CHUNK_SIZE_TOKEN_OVERFLOW.md** - 🟡 Proposed (hotfix + proper fix pending)
- **EMBEDDING_THROUGHPUT_INVESTIGATION.md** - ✅ Complete (pre-truncation implemented; ONNX/quantisation deferred)

## Purpose

Proposals explore design options before making decisions. Once a proposal is accepted, it becomes a decision (see [../decisions/](../decisions/)).
