# ArionXiv

A command-line interface for discovering, analyzing, and interacting with research papers from arXiv.

---

## Installation

```bash
pip install arionxiv
```

If the command is not found after installation, add Python scripts to PATH:

**Windows (PowerShell):**
```powershell
python -c "import sysconfig; p=sysconfig.get_path('scripts'); import os; os.system(f'setx PATH \"%PATH%;{p}\"')"
```

**macOS / Linux:**
```bash
echo "export PATH=\"\$PATH:$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"scripts\"))')\"" >> ~/.bashrc && source ~/.bashrc
```

---

## Getting Started

### First Run

```bash
arionxiv
```

On first run, register or login to your account:

```bash
arionxiv register   # Create a new account
arionxiv login      # Login to existing account
```

That's it. No API keys or configuration required.

---

## Features

### 1. Paper Search

Search arXiv with relevance scoring and filtering.

```bash
arionxiv search "transformer architecture"
arionxiv search "reinforcement learning" --max-results 20
```

<!-- Screenshot: search results -->

---

### 2. Paper Analysis

AI-powered deep analysis of research papers.

```bash
arionxiv analyze 2301.00001
arionxiv analyze 2301.00001 --detailed
```

<!-- Screenshot: analysis output -->

---

### 3. Chat with Papers

Interactive RAG-based Q&A with any paper. Supports session persistence and history.

```bash
arionxiv chat
arionxiv chat 2301.00001
```

**Features:**
- Context-aware responses using paper content
- Session persistence across restarts
- Chat history (last 8 Q&A pairs) on resume
- Cached embeddings for instant session loading

<!-- Screenshot: chat interface -->

---

### 4. Personal Library

Save papers and manage your research collection.

```bash
arionxiv library
arionxiv settings papers
```

<!-- Screenshot: library view -->

---

### 5. Daily Dose

Personalized daily paper recommendations based on your research interests.

```bash
arionxiv daily
arionxiv daily --run
arionxiv daily --view
```

Configure schedule and preferences:

```bash
arionxiv settings daily
```

<!-- Screenshot: daily dose -->

---

### 6. Trending Papers

Discover trending research topics and papers.

```bash
arionxiv trending
```

<!-- Screenshot: trending view -->

---

### 7. Themes

Customizable terminal interface with multiple color themes.

```bash
arionxiv settings theme
```

Available themes: cyan, green, magenta, yellow, red, blue, white

<!-- Screenshot: theme options -->

---

## Command Reference

| Command | Description |
|---------|-------------|
| `arionxiv` | Main menu |
| `arionxiv search <query>` | Search for papers |
| `arionxiv fetch <paper_id>` | Download paper PDF |
| `arionxiv analyze <paper_id>` | AI analysis |
| `arionxiv chat [paper_id]` | Chat with papers |
| `arionxiv daily` | Daily recommendations |
| `arionxiv trending` | Trending topics |
| `arionxiv library` | Saved papers |
| `arionxiv settings` | Configuration |
| `arionxiv login` | Authenticate |
| `arionxiv register` | Create account |
| `arionxiv session` | Check auth status |
| `arionxiv --help` | Show all commands |

---

## Configuration

### Settings Commands

```bash
arionxiv settings show      # View all settings
arionxiv settings theme     # Change color theme
arionxiv settings api       # Configure optional API keys (Gemini, Groq, HuggingFace)
arionxiv settings prefs     # Research preferences
arionxiv settings daily     # Daily dose schedule
arionxiv settings papers    # Manage saved papers
```

### Self-Hosting (Optional)

If you want to run your own backend instead of using the hosted service:

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB connection string |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `JWT_SECRET_KEY` | Authentication secret |
| `GEMINI_API_KEY` | Google Gemini embeddings (optional) |
| `GROQ_API_KEY` | Fallback LLM provider (optional) |

---

## Optional Dependencies

```bash
pip install arionxiv[advanced-pdf]  # OCR and table extraction
pip install arionxiv[ml]            # Local embeddings
pip install arionxiv[all]           # All extras
```

---

## Daily Dose Automation

### GitHub Actions

1. Fork the repository
2. Add secrets in Settings > Secrets:
   - `MONGODB_URI`
   - `OPENROUTER_API_KEY`
   - `JWT_SECRET_KEY`
3. The workflow runs hourly and processes users based on their scheduled time

---

## Alternative Invocation

If the `arionxiv` command is not available:

```bash
python -m arionxiv <command>
```

---

## Links

- PyPI: https://pypi.org/project/arionxiv/
- GitHub: https://github.com/ArionDas/ArionXiv
- Issues: https://github.com/ArionDas/ArionXiv/issues

---

## License

MIT License
