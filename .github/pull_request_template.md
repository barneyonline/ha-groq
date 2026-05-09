## Summary

Describe the change and why it is needed.

## Testing

List the exact commands run. Prefer the Docker test environment:

```bash
scripts/test
scripts/test python -m pytest tests/components/groq -q
```

## Checklist

- [ ] I updated documentation for user-facing changes.
- [ ] I added or updated tests for changed behavior.
- [ ] I verified the integration still imports in the Home Assistant test environment.
