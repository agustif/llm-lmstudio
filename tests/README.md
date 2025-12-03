# Tests

This directory contains test files for the llm-lmstudio plugin.

## Test Files

### Core Plugin Tests

- `test_sync.py` - Exercises synchronous message building, attachments, and tool handling
- `test_async.py` - Covers async execution helpers and streaming support

### Integration Tests  

- `test_end_to_end.py` - Smoke tests that run the plugin through llm’s end-to-end interface

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test files

```bash
pytest tests/test_sync.py
pytest tests/test_async.py
pytest tests/test_end_to_end.py
```



## Requirements

- LM Studio running on localhost:1234 (for integration tests)
- Compatible models loaded for specific tests
- pytest installed (`pip install pytest`)

## Test Data

The `cassettes/` directory contains VCR cassettes for recorded HTTP interactions used in some tests.
