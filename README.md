# Q&A Application

An asynchronous command-line application that leverages the Perplexity API to generate comprehensive insights by analyzing questions and providing related follow-up questions with answers.

## Features

- Utilizes Perplexity API for intelligent question analysis
- Asynchronous processing for efficient handling of multiple questions
- Generates related follow-up questions for deeper insights
- Saves results in both JSON and Markdown formats
- Interactive CLI with progress animations
- Structured output with clear formatting
- Secure API key handling through environment variables

## Prerequisites

- Python 3.10+
- Perplexity API key
- Poetry (for dependency management)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd questions
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Create a `.env` file in the project root with your Perplexity API key:

```bash
PERPLEXITY_API_KEY=your_api_key_here
```

## Usage

### Running the Application

```bash
poetry run python -m questions.main
```

The application will:

1. Prompt you to specify if you want to log output to a file
2. Ask for your question
3. Request the type of expert perspective you want
4. Generate and process related questions
5. Display answers and save results

### Programmatic Usage

You can also run the application with pre-defined parameters:

```python
import asyncio
from questions.main import main

asyncio.run(main(
    question="What are the key factors in retirement planning?",
    expert_type="financial advisor",
    log_to_file=True
))
```

### Available Expert Types

The application includes prompt templates for several expert domains:

- `financial_advisor` - Financial planning and advice
- `tax_expert` - Tax-related questions
- `investment_analyst` - Investment analysis
- `business_consultant` - Business strategy
- `mill_research` - Textile industry research

## Configuration

The application uses an immutable `ConfigTuple` for settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `output_dir` | `results/` | Directory for saving results |
| `log_file` | `app.log` | Log file location |
| `model_name` | `llama-3.1-sonar-large-128k-online` | Perplexity model |
| `base_url` | `https://api.perplexity.ai` | API base URL |

## Output Formats

Results are saved to the `results/` directory with timestamp-based filenames.

### JSON Output

```json
{
  "original_question": "user's question",
  "timestamp": "YYYYMMDD_HHMMSS",
  "results": [
    {
      "question": "question text",
      "answer": "answer text"
    }
  ]
}
```

### Markdown Output

Results are also saved in a formatted Markdown file including:

- Timestamp
- Original question
- List of all questions and answers
- Clear section formatting

## Development

### Running Tests

```bash
poetry run pytest
```

Or using the task runner:

```bash
poetry run invoke test
```

### Linting and Formatting

```bash
poetry run invoke lint    # Run flake8 and mypy
poetry run invoke format  # Run black and isort
```

### Generate Documentation

```bash
poetry run pdoc --output-dir docs questions
```

### Run All Checks

```bash
poetry run invoke all     # format, lint, test
```

## Project Structure

```
questions/
├── questions/              # Main package
│   ├── __init__.py
│   ├── main.py            # Core pipeline and entry point
│   ├── config.py          # Configuration management
│   ├── prompts.py         # Expert prompt templates
│   ├── io_utils.py        # User input/output handling
│   ├── file_utils.py      # File operations
│   └── animation_utils.py # CLI progress animations
├── tests/                 # Test suite
├── results/               # Output directory
├── pyproject.toml         # Poetry configuration
└── tasks.py               # Invoke task automation
```

## Error Handling

The application includes comprehensive error handling for:

- Missing API keys
- Invalid user input
- API communication errors
- Keyboard interrupts

## Acknowledgments

- Built with the Perplexity API
- Uses OpenAI's Python client library
- Async support through asyncio
