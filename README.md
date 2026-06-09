# CZ Certification Automation

Automated certification testing framework for CZ (Central Zone/Portal) environments.

## Features

- **Web Scraping**: Automated scraping of CZ portal test cases with Playwright
- **Dependency Management**: DAG-based dependency resolution with automatic skip logic
- **LLM-Driven Execution**: LiteLLM-powered payload generation and self-correction
- **Dual Validation**: UI status verification and pod log analysis
- **Human Review**: Web dashboard and CLI fallback for command approval
- **Comprehensive Reporting**: HTML dashboard with dependency visualization

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required settings:
- `CZ_BASE_URL`: CZ portal URL
- `JSESSIONID`: Session cookie (or leave empty to be prompted)
- `LITELLM_*`: LLM configuration
- `K8S_*`: Kubernetes pod details

## Usage

### Basic Execution

```bash
python main.py
```

### With Repository Context

```bash
python main.py --repo-path /path/to/your/app
```

### Parallel Execution

```bash
python main.py --parallel --workers 8
```

### Human Review Mode

```bash
python main.py --human-review
```

### Dry Run

```bash
python main.py --dry-run
```

## Project Structure

```
cz_automation/
├── config/           # Configuration and settings
├── core/            # Core models, DAG engine, state machine
├── scraper/         # Playwright-based web scraper
├── llm_agent/       # LiteLLM client and prompt builder
├── validator/       # UI and log validation
├── reporting/       # HTML dashboard generation
├── utils/           # Helper utilities and dashboard server
└── main.py          # CLI entrypoint
```

## License

MIT
