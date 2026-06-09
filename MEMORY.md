# CZ Certification Automation - Memory & Context Document

**Version:** 0.1.0  
**Last Updated:** 2026-06-09  
**Purpose:** Comprehensive reference for future sessions to understand architecture, design decisions, and implementation details without re-reading the entire codebase.

---

## 1. Executive Summary

This is an **automated certification framework** for CZ (Central Zone/Portal) environments. It automates the execution and validation of certification test cases (TCs) hosted on the CZ portal through three interaction patterns:

1. **Outbound API Calls**: System initiates API request to CZ portal
2. **Chained API Calls**: Parse response, use data for subsequent API call
3. **Inbound Webhook**: Click "Play" on portal, CZ sends request to our system, we respond

### Key Achievement
Complete Python monolith (3,662 LOC) implementing full automation pipeline from web scraping to LLM-driven payload generation to HTML reporting.

---

## 2. Architecture Overview

```
CLI Entrypoint (main.py)
    │
    ├── Configuration (Pydantic Settings + .env)
    │
    ├── State Machine (core/state_machine.py)
    │       ├── INIT
    │       ├── SCRAPING (Phase 1)
    │       ├── PORT_FORWARD_SETUP
    │       ├── EXECUTING (Phase 2)
    │       ├── VALIDATING (Phase 3)
    │       ├── REPORTING
    │       └── DONE
    │
    ├── Scraper Module (scraper/)
    │       ├── Browser Manager (cookie injection, anti-detection)
    │       ├── Navigator (pagination, SSR handling)
    │       ├── Extractor (DOM parsing, dependency scraping)
    │       ├── Trigger (JS event firing, Play button clicking)
    │       └── Auditor (screenshot lifecycle)
    │
    ├── LLM Agent (llm_agent/)
    │       ├── LiteLLM Client (dynamic model selection)
    │       ├── Context Loader (scoped repo scanning)
    │       ├── Prompt Builder (initial + correction + chained)
    │       └── Command Executor (curl execution)
    │
    ├── Validator (validator/)
    │       ├── UI Validator (green tick / red cross detection)
    │       ├── Pod Log Fetcher (kubectl logs | grep correlation_id)
    │       └── Categorizer (TIMEOUT, SCHEMA, RESPONSE, NETWORK)
    │
    ├── K8s Bridge (core/k8s_bridge.py)
    │       ├── Port Forward Manager (health check + auto-restart)
    │       └── Log Fetcher (filtered by correlation IDs)
    │
    ├── DAG Engine (core/dag_engine.py)
    │       ├── Topological ordering
    │       ├── Wave-based parallel scheduling
    │       └── Recursive dependency skip on failure
    │
    └── Reporting (reporting/)
            ├── HTML Dashboard Generator
            ├── Jinja2 Templates with Mermaid.js graphs
            └── Dashboard Server (aiohttp, optional human review panel)
```

---

## 3. Design Decisions & Rationale

### 3.1 Tech Stack
- **Python 3.10+**: Chosen for async/await, type hints, ecosystem maturity
- **Playwright**: Selected over Selenium for reliable JS event listener handling and native async support
- **LiteLLM**: Universal LLM proxy allowing model hot-swapping without code changes
- **NetworkX**: Industry-standard graph library for DAG operations
- **Pydantic Settings**: Type-safe configuration with .env fallback
- **Rich**: Beautiful CLI output for human-in-the-loop interactions

### 3.2 Monolithic Architecture
Single repository with logical module separation. NOT microservices because:
- Runs locally on developer machine
- Tight coupling between phases (scrape → execute → validate)
- Simplicity outweighs deployment flexibility

### 3.3 Authentication Strategy
- **Primary**: CLI prompt at runtime for JSESSIONID
- **Fallback**: `.env` file for unattended runs
- **Session Management**: Injected as cookie into Playwright browser context
- **TTL Handling**: Manual re-auth when session expires (currently no auto-refresh)

### 3.4 Parallel Execution Strategy
**Wave-based scheduling** (NOT naive parallel):
- Build DAG during scraping
- Group test cases into "waves" where all dependencies are satisfied
- Execute each wave with semaphore-limited concurrency
- Safe and deterministic compared to dynamic task spawning

### 3.5 Human Review Gate
Two-tier approach:
1. **Primary**: Web dashboard (aiohttp server on localhost:8765) with Approve/Reject/Edit buttons
2. **Fallback**: CLI prompt (`Enter`=approve, `r`=reject, `e`=edit)
- Never hangs blindly - always has fallback path
- Configurable via `--human-review` flag

### 3.6 LLM Context Management
- **Scoped Loading**: Only scans `allowed_context_paths` (default: schemas/, fixtures/, templates/)
- **Token Budgeting**: Uses `tiktoken` to estimate and trim context to ~80k tokens
- **Priority Ordering**: XSD > XML > JSON > YAML (most structurally important first)
- **Truncation**: If context exceeds budget, truncates lower-priority files

### 3.7 Self-Correction Loop
```
Initial Generation → Execute → Error?
    ↓
Feed Error + Logs + Previous Command to LLM
    ↓
Generate Corrected Command → Execute
    ↓
Repeat up to MAX_LLM_RETRIES (default: 3)
    ↓
If still failing → Mark FAILED → Skip Dependents → Continue
```

### 3.8 Port-Forward Resilience
- Health check before EVERY test case execution
- Probes `http://localhost:8080/health` (configurable)
- Auto-restarts `kubectl port-forward` subprocess if stale
- Cleans up existing processes on same port before starting

### 3.9 Log Filtering Strategy
**CRITICAL**: Never pass raw `kubectl logs` output to LLM
- Extract correlation IDs during scraping/execution
- Filter: `kubectl logs <pod> --tail=500 | grep <correlation_id>`
- Only filtered, relevant logs sent to LLM for analysis
- Reduces noise and token consumption

---

## 4. Configuration Schema

### Environment Variables (via `.env` or system env)

```env
# CZ Portal
CZ_BASE_URL=https://portal.example.com/TestCases/List2
JSESSIONID=your-session-cookie

# Repository Context
REPO_PATH=/path/to/app
ALLOWED_CONTEXT_PATHS=schemas/,fixtures/,templates/

# LiteLLM (dynamic - can override via CLI or dashboard)
LITELLM_BASE_URL=http://localhost:4000
LITELLM_MODEL=gpt-4o
LITELLM_API_KEY=your-key

# Kubernetes
K8S_POD=my-pod
K8S_NAMESPACE=my-namespace
K8S_LOCAL_PORT=8080
K8S_REMOTE_PORT=8080
K8S_HEALTH_CHECK_ENDPOINT=http://localhost:8080/health

# Execution Control
RATE_LIMIT_DELAY_MS=1000
MAX_LLM_RETRIES=3
MAX_TC_EXECUTION_TIMEOUT_SEC=60
PARALLEL=false
MAX_PARALLEL_WORKERS=4
HUMAN_REVIEW=false

# Output
ARTIFACTS_DIR=./artifacts
```

### CLI Overrides
```bash
python main.py \
  --repo-path /path/to/app \
  --parallel \
  --workers 8 \
  --human-review \
  --model claude-3-sonnet-20240229 \
  --dry-run
```

---

## 5. Data Models

### TestCase (core/models.py)
```python
@dataclass
class TestCase:
    id: str                           # Unique identifier (e.g., "TC-001")
    description: str                  # Scraped from portal
    expected_status: str              # Expected outcome
    payload_requirements: Optional[str]
    dependencies: List[str]           # TC IDs this depends on
    play_button_selector: Optional[str]
    status: TestStatus               # PENDING → RUNNING → PASSED/FAILED/SKIPPED
    failure_category: Optional[FailureCategory]
    screenshots: Dict[str, Path]     # before/during/after/failure
    pod_logs: Optional[str]          # Filtered kubectl logs
    portal_logs: Optional[str]       # Clicked from portal UI
    llm_attempts: List[dict]         # Retry history
    correlation_ids: List[str]       # Extracted transaction IDs
    generated_curl_command: Optional[str]
    execution_output: Optional[str]
    execution_error: Optional[str]
```

### TestSuiteResult (core/models.py)
```python
@dataclass
class TestSuiteResult:
    total: int
    passed: int
    failed: int
    skipped: int
    pending: int
    test_cases: List[TestCase]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
```

### Enums (core/enums.py)
- `TestStatus`: PENDING, RUNNING, PASSED, FAILED, SKIPPED
- `FailureCategory`: TIMEOUT, SCHEMA_VALIDATION, INCORRECT_RESPONSE, NETWORK_ERROR, UNKNOWN
- `ExecutionMode`: SEQUENTIAL, PARALLEL
- `ReviewAction`: APPROVE, REJECT, EDIT

---

## 6. File Structure

```
cz_automation/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic Settings, env vars
│
├── core/
│   ├── __init__.py
│   ├── dag_engine.py        # networkx DiGraph, topological sort, waves
│   ├── enums.py             # Status, Category, Mode enums
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── k8s_bridge.py        # PortForwardManager, LogFetcher
│   ├── models.py            # TestCase, TestSuiteResult dataclasses
│   └── state_machine.py     # Orchestrator, phase transitions
│
├── llm_agent/
│   ├── __init__.py
│   ├── client.py            # LiteLLM wrapper with retry
│   ├── context_loader.py    # Scoped repo scanner, token budgeting
│   ├── executor.py          # subprocess curl execution
│   └── prompt_builder.py    # Initial, Correction, Chained prompts
│
├── scraper/
│   ├── __init__.py
│   ├── auditor.py           # Screenshot lifecycle
│   ├── browser.py           # Playwright context, cookie injection
│   ├── extractor.py         # DOM parsing, dependency extraction
│   ├── navigator.py         # Pagination, SSR handling
│   ├── orchestrator.py      # Main scraper coordinator
│   └── trigger.py           # Play button JS event firing
│
├── validator/
│   ├── __init__.py
│   └── ui_validator.py      # UI polling, categorizer
│
├── reporting/
│   ├── __init__.py
│   ├── dashboard.py         # HTML generator
│   └── templates/
│       └── dashboard.html   # Jinja2 + Mermaid.js
│
├── utils/
│   ├── __init__.py
│   ├── dashboard_server.py  # aiohttp server for web review
│   ├── helpers.py           # Utility functions
│   └── human_review.py      # Review gate (web + CLI)
│
├── tests/
│   └── __init__.py
│
├── artifacts/               # Created at runtime
│   ├── screenshots/         # Audit trail images
│   └── logs/               # Execution logs
│
├── .env.example            # Template configuration
├── pyproject.toml          # Package metadata, dependencies
├── requirements.txt        # Pip dependencies
├── README.md               # Quick start guide
├── main.py                 # CLI entrypoint (Click + Rich)
└── MEMORY.md               # This document
```

---

## 7. Execution Flow

### Sequential Mode
```
1. Initialize (load config, create dirs)
2. Start browser, inject JSESSIONID
3. Navigate to CZ portal test suite page
4. FOR EACH page:
    a. Extract test case metadata (ID, desc, status, deps)
    b. Navigate to next page
5. Build DAG from all test cases
6. Start kubectl port-forward
7. Health check port-forward
8. FOR EACH test case IN topological order:
    a. IF blocked by failed dependency → SKIP
    b. Capture "before" screenshot
    c. Click Play button (JS event)
    d. Capture "during" screenshot
    e. IF outbound/chained API needed:
        - Load repo context (XSD/XML/JSON)
        - Build LLM prompt
        - Human review (if enabled)
        - Generate curl command
        - Execute curl
        - IF error AND retries left:
            → Feed error to LLM for correction
            → Retry
    f. Wait for portal execution (max 60s)
    g. Validate UI status (green tick / red cross)
    h. IF FAILED:
        - Click Logs icon
        - Fetch filtered pod logs
        - Categorize failure
        - Mark FAILED
        - Skip all dependents recursively
    i. ELSE:
        - Mark PASSED
    j. Capture "after" screenshot
    k. Rate limit delay
9. Generate HTML dashboard report
10. Cleanup (close browser, stop port-forward)
```

### Parallel Mode (Wave-Based)
```
Same as sequential BUT step 8 becomes:

8. Group test cases into waves (DAG depth levels)
   FOR EACH wave:
       Execute all test cases concurrently (semaphore limited)
       Wait for all to complete
       Update failed_nodes set
       Next wave respects updated blocked status
```

---

## 8. Key Algorithms

### 8.1 DAG Wave Scheduling
```python
waves = []
executed = set()

while len(executed) < graph.number_of_nodes():
    wave = [
        node for node in graph.nodes()
        if node not in executed
        and all(pred in executed for pred in graph.predecessors(node))
    ]
    waves.append(wave)
    executed.update(wave)
```

### 8.2 Recursive Dependency Skip
```python
def skip_subtree(failed_tc_id):
    skipped = []
    
    def _mark_skipped(node_id):
        tc = node_map[node_id]
        if tc.status == PENDING:
            tc.status = SKIPPED
            skipped.append(node_id)
        for dependent in graph.successors(node_id):
            _mark_skipped(dependent)
    
    _mark_skipped(failed_tc_id)
    return skipped
```

### 8.3 Port-Forward Health Check
```python
async def health_check():
    if not process or process.poll() is not None:
        return False
    try:
        response = await httpx.get(health_endpoint, timeout=5)
        return response.status_code < 500
    except:
        return False

async def restart_if_stale():
    if not await health_check():
        stop()
        start()
        await sleep(2)
```

### 8.4 Log Filtering by Correlation ID
```python
raw_logs = subprocess.run(
    ["kubectl", "logs", pod, "--tail=500"],
    capture_output=True
).stdout

filtered = "\n".join(
    line for line in raw_logs.split("\n")
    if any(cid in line for cid in correlation_ids)
)
```

---

## 9. Prompt Engineering Strategy

### System Prompt (Initial Generation)
```
You are an expert certification automation engineer...
RULES:
1. Analyze test case description carefully
2. Reference provided XML schemas and sample payloads
3. Generate exact curl commands with correct headers, namespaces, content types
4. Include all necessary correlation IDs
5. Handle positive and negative scenarios
6. ALWAYS return curl command in code block
```

### System Prompt (Correction)
```
You are correcting a previously generated curl command that failed...
CORRECTION GUIDELINES:
1. Fix specific error indicated
2. Ensure XML schema compliance
3. Verify correlation IDs placement
4. Double-check headers and namespaces
```

### Context Injection Format
```
=== REFERENCE DATA FROM APPLICATION REPOSITORY ===
--- FILE: schemas/billing.xsd ---
[content]
--- FILE: fixtures/sample_request.xml ---
[content]
=== END REFERENCE DATA ===
```

---

## 10. Known Limitations & Future Improvements

### Current Limitations
1. **No Session Refresh**: If JSESSIONID expires mid-run, automation fails. User must restart with fresh cookie.
2. **Hardcoded Selectors**: Playwright selectors are educated guesses. Actual portal DOM may require adjustment.
3. **No Web Dashboard Yet**: `DashboardServer` scaffolding exists but `aiohttp` review endpoints need implementation.
4. **Pod Name Discovery**: Currently static. Could auto-discover via `kubectl get pods` label selectors.
5. **LLM Token Limits**: Very large repos may still exceed context windows despite trimming.

### Planned Enhancements
1. **Automatic Session Refresh**: Detect expired sessions and pause for re-auth
2. **DOM Snapshot Debugging**: Save full page HTML on failures for analysis
3. **Metrics Export**: Prometheus-compatible metrics endpoint
4. **CI/CD Integration**: GitHub Actions / Jenkins pipeline wrappers
5. **Test Case Diffing**: Compare scraped TCs with previous runs to detect portal changes
6. **Smart Wait Strategies**: Replace fixed sleeps with mutation observers

---

## 11. Testing Strategy

### Unit Tests (TODO)
- `test_dag_engine.py`: Topological sort, wave generation, skip logic
- `test_context_loader.py`: Token estimation, trimming, priority ordering
- `test_prompt_builder.py`: Prompt construction, context injection
- `test_categorizer.py`: Failure classification accuracy

### Integration Tests (TODO)
- `test_scraper_flow.py`: Mock HTML page, verify extraction
- `test_k8s_bridge.py`: Mock kubectl responses
- `test_state_machine.py`: Full lifecycle with mock components

### Manual Testing Checklist
- [ ] Browser launches with cookie injection
- [ ] Navigates to test suite page
- [ ] Extracts test cases with dependencies
- [ ] Clicks Play button successfully
- [ ] Takes screenshots at all phases
- [ ] Port-forward starts and health-checks
- [ ] LLM generates valid curl commands
- [ ] Self-correction loop works
- [ ] HTML report generates with graphs
- [ ] Human review CLI fallback responds

---

## 12. Troubleshooting Guide

### Symptom: Browser shows login page instead of test suite
**Cause**: JSESSIONID expired or incorrect
**Fix**: Run `python main.py` and enter fresh cookie after manual login

### Symptom: Port-forward dies randomly
**Cause**: VPN instability or idle timeout
**Fix**: Health check auto-restarts; if persistent, check VPN connection

### Symptom: LLM generates invalid XML
**Cause**: Insufficient context or model confusion
**Fix**: Increase `allowed_context_paths`, use more capable model, add more sample XML to repo

### Symptom: Test cases show as failed despite correct payload
**Cause**: Correlation ID mismatch or callback not received
**Fix**: Check `kubectl logs` manually, verify callback endpoint configuration in app

### Symptom: Scraper can't find Play button
**Cause**: DOM selectors don't match actual portal
**Fix**: Inspect portal HTML, update selectors in `scraper/extractor.py`

---

## 13. Dependencies

### Production
- `playwright>=1.40`: Browser automation
- `litellm>=1.0`: LLM proxy
- `pydantic>=2.0` + `pydantic-settings>=2.0`: Configuration
- `jinja2>=3.1`: Template engine
- `networkx>=3.0`: Graph algorithms
- `click>=8.0`: CLI framework
- `rich>=13.0`: Terminal UI
- `httpx>=0.25`: HTTP client
- `tiktoken>=0.5`: Token counting

### Development
- `pytest>=7.0` + `pytest-asyncio>=0.21`: Testing
- `black>=23.0`: Code formatting
- `flake8>=6.0`: Linting
- `mypy>=1.0`: Type checking

---

## 14. Extension Points

### Adding New Failure Categories
Edit `core/enums.py::FailureCategory`, then update `validator/ui_validator.py::Categorizer`

### Supporting Different LLM Providers
Already supported via LiteLLM. Change `LITELLM_MODEL` env var. Examples:
- OpenAI: `gpt-4o`, `gpt-4-turbo`
- Anthropic: `claude-3-opus-20240229`
- Local: `ollama/llama3`

### Custom Prompt Templates
Extend `llm_agent/prompt_builder.py::PromptBuilder` with new methods

### Alternative Report Formats
Extend `reporting/dashboard.py` or add new generators for JSON, CSV, JUnit XML

---

## 15. Session Continuation Notes

When resuming work on this project:

1. **Read this MEMORY.md first** for context
2. **Check `main.py`** for CLI interface
3. **Review `core/state_machine.py`** for execution flow
4. **Examine `.env.example`** for configuration options
5. **Look at `scraper/extractor.py`** - most likely to need tuning for actual portal DOM
6. **Test `llm_agent/prompt_builder.py`** - may need prompt engineering for specific models

The most critical files for understanding system behavior:
- `core/state_machine.py` (orchestration)
- `scraper/orchestrator.py` (Phase 1)
- `core/k8s_bridge.py` (infrastructure)
- `llm_agent/client.py` (AI integration)
- `validator/__init__.py` (validation logic)

---

## 16. Contact & References

- **GitHub Issues**: https://github.com/anomalyco/opencode/issues (for opencode tool feedback)
- **CZ Portal URL**: https://portal.hsbcacq.ibmb.uat.in1.juspay.in/TestCases/List2 (UAT)
- **Tech Stack**: Python 3.10+, Playwright, LiteLLM, NetworkX

---

**End of Memory Document**

This document serves as the single source of truth for architecture decisions, implementation details, and operational procedures. Update it when making significant changes to the framework.
