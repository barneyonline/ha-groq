---
name: ha-groq-startup
description: Start and verify the ha-groq managed Codex Cloud environment once per new chat before repository development. Use only in this repository's managed cloud environment.
---

# ha-groq cloud startup

Before development in a new managed cloud chat, read
`/workspace/.agents/skills/ha-groq-startup/SKILL.md`
and follow its existing startup and readiness workflow once in this chat.
That environment-owned file is authoritative for commands and tool versions.

Require successful readiness before editing or running development checks.
If the file or its helpers are missing, report the missing setup and stop;
do not substitute host Python, install tools, or invent a startup command.
Preserve the checkout, caches, managed credentials, proxy and CA configuration.
