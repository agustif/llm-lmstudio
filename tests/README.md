# Tests

This directory contains test files for the llm-lmstudio plugin.

## Test Files

### Core Plugin Tests

- `test_llm_lmstudio.py` - Basic sync functionality tests
- `test_llm_lmstudio_async.py` - Async functionality tests

### Enhanced Features Tests  

- `test_thinking_analysis.py` - Tests for thinking tag handling with qwen models
- `test_timeout_handling.py` - Tests for enhanced timeout handling with tools

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test files

```bash
pytest tests/test_llm_lmstudio.py
pytest tests/test_thinking_analysis.py
```

### Run with timeout testing

```bash
# Set timeout for timeout handling tests
LMSTUDIO_TIMEOUT=15 pytest tests/test_timeout_handling.py
```

## Requirements

- LM Studio running on localhost:1234 (for integration tests)
- Compatible models loaded for specific tests
- pytest installed (`pip install pytest`)

## Test Data

The `cassettes/` directory contains VCR cassettes for recorded HTTP interactions used in some tests.
