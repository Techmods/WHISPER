# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a Claude Code agent configuration repository for a German-language Windows AI setup assistant. It contains no build artifacts, source code, or dependencies — only agent definitions and their persistent memory.

## Structure

```
.claude/
  agents/
    windows-ai-setup-de.md     # Agent definition (model: sonnet, memory: project)
  agent-memory/
    windows-ai-setup-de/       # Persistent memory store for the agent (MEMORY.md + memory files)
```

## The `windows-ai-setup-de` Agent

Defined in [.claude/agents/windows-ai-setup-de.md](.claude/agents/windows-ai-setup-de.md). It:

- Responds **exclusively in German**
- Helps users install and configure a **3-agent hierarchical AI system on Windows**:
  - Agent 1 (Haiku): research and documentation tasks
  - Agent 2 (Sonnet): programming tasks
  - Agent 3: orchestration and planning
- Structures every response with `<scratchpad>`, `<recherche>`, `<architektur>`, `<installationsplan>`, and `<hinweise>` sections
- Writes Windows-specific guidance using PowerShell, and addresses Windows Defender, PATH, UTF-8 encoding, and WSL2 considerations

## Agent Memory System

The agent uses a file-based project-scoped memory at `.claude/agent-memory/windows-ai-setup-de/`. Memories are indexed in a `MEMORY.md` file inside that directory. The agent saves recurring Windows compatibility patterns, Claude API configuration patterns, and multi-agent architecture decisions across conversations.

When modifying the agent definition, preserve the memory system section at the bottom of the file — it instructs the agent how and when to persist institutional knowledge.
