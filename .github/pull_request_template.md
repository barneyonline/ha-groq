## Summary

Describe the change and why it is needed.

## Related Issues

List related issues or discussions. Use `Fixes #123` only when this PR closes the issue.

## Type of change

- [ ] Bugfix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / tech debt
- [ ] Translation update
- [ ] Repository tooling / CI
- [ ] Other

## Testing

List the exact commands run. Prefer the Docker test environment and keep commands scoped when possible:

```bash
scripts/test
scripts/test python -m pytest tests/components/groq -q
scripts/test ruff check custom_components/groq tests/components/groq tests/scripts scripts
scripts/strict-typing
scripts/test python scripts/validate_quality_scale.py
scripts/test python scripts/importtime_profile.py --strict-integration-warnings
```

## Checklist

- [ ] I updated `README.md` or `CONTRIBUTING.md` for user-facing or contributor-facing changes.
- [ ] I updated `custom_components/groq/services.yaml` when action inputs or response shapes changed.
- [ ] I updated translations under `custom_components/groq/translations/` when strings changed.
- [ ] I added or updated tests for changed behavior.
- [ ] I ran the relevant Docker-backed checks and listed them above.
- [ ] I updated `quality_scale.yaml` when quality-scale evidence changed.
- [ ] I verified diagnostics, repairs, and errors do not expose Groq API keys or private prompt/media content.
- [ ] I considered whether the manifest version, HACS metadata, dependencies, loggers, or quality scale claim need updates.

## Diagnostics / Screenshots / Notes

Add screenshots, redacted diagnostics, repair issue context, or implementation notes reviewers should see.
