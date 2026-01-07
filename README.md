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

### First Run - Start with Welcome

**The first command you should always run is:**

```bash
arionxiv welcome
```

This will display the welcome screen and guide you through the initial setup.

### Create an Account

**First-time users must create an account to use ArionXiv:**

```bash
arionxiv register   # Create a new account (required for first-time users)
arionxiv login      # Login to existing account
```

Once registered, you can access all features including paper search, AI analysis, chat, and personalized recommendations.

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

AI-powered deep analysis of research papers. Access this feature by searching for a paper and selecting "Analyze" from the results menu.

```bash
arionxiv search "transformer architecture"
# Select a paper → Choose "Analyze"
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
| `arionxiv welcome` | Welcome screen (run this first!) |
| `arionxiv` | Main menu |
| `arionxiv search <query>` | Search for papers (with analyze option) |
| `arionxiv chat [paper_id]` | Interactive RAG chat with papers |
| `arionxiv daily` | Daily personalized recommendations |
| `arionxiv trending` | Discover trending topics |
| `arionxiv library` | View saved papers |
| `arionxiv settings` | Configuration menu |
| `arionxiv register` | Create new account |
| `arionxiv login` | Login to existing account |
| `arionxiv session` | Check authentication status |
| `arionxiv --help` | Show all commands |

---

## Configuration

### Settings Commands

```bash
arionxiv settings show         # View all settings
arionxiv settings theme        # Change color theme
arionxiv settings api          # Configure optional API keys (Gemini, Groq, HuggingFace)
arionxiv settings preferences  # Research preferences
arionxiv settings daily        # Daily dose schedule
arionxiv settings papers       # Manage saved papers
```

### Self-Hosting (For Developers Only)

> **Note:** Regular users do NOT need to self-host. ArionXiv automatically connects to our hosted backend service. This section is only for developers who want to run their own backend infrastructure.

If you want to use your own APIs and LLM providers, set the following using command "arionxiv settings api":

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM |
| `JWT_SECRET_KEY` | Authentication secret |
| `GEMINI_API_KEY` | Google Gemini embeddings (optional) |
| `GROQ_API_KEY` | Fallback LLM provider (optional) |

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
