# ArionXiv - AI-Powered Research Paper Analysis

**ArionXiv** is a CLI tool for discovering, analyzing, and managing research papers from arXiv with AI-powered insights.

## Features

- **Smart Search**: Find papers with intelligent relevance scoring
- **AI Analysis**: Deep paper analysis using OpenRouter (FREE models available)
- **Daily Dose**: Personalized daily paper recommendations
- **Chat with Papers**: Ask questions about any paper
- **Personal Library**: Organize your research collection
- **Beautiful CLI**: Rich terminal interface with customizable themes
- **Trending**: Discover trending research topics

## Installation

```bash
pip install arionxiv
```

### Command Not Found?

If `arionxiv` command is not found, run this to fix PATH and restart your terminal:

**Windows (PowerShell):**
```powershell
python -c "import sysconfig; p=sysconfig.get_path('scripts'); import os; os.system(f'setx PATH \"%PATH%;{p}\"')"
```

**Windows (Command Prompt):**
```cmd
for /f "tokens=*" %i in ('python -c "import sysconfig; print(sysconfig.get_path('scripts'))"') do setx PATH "%PATH%;%i"
```

**macOS / Linux:**
```bash
echo "export PATH=\"\$PATH:$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"scripts\"))')\"" >> ~/.bashrc && source ~/.bashrc
```

## Quick Start

### 1. First Run Setup

On first run, ArionXiv will guide you through API key setup:

```bash
arionxiv
```

### 2. Required Setup

You need:
- **MongoDB Atlas** (free tier available): https://www.mongodb.com/atlas
- **OpenRouter API Key** (FREE): https://openrouter.ai/keys

### 3. Basic Commands

```bash
# Search for papers
arionxiv search "transformer architecture"

# Fetch a paper
arionxiv fetch 2301.00001

# Analyze a paper with AI
arionxiv analyze 2301.00001

# Chat with a paper
arionxiv chat 2301.00001

# Get your daily dose
arionxiv daily

# See trending topics
arionxiv trending

# Manage settings
arionxiv settings
```

> **Alternative**: Replace `arionxiv` with `python -m arionxiv` if the command is not found.

### 4. Authentication

```bash
# Register a new account
arionxiv register

# Login
arionxiv login

# Check session
arionxiv session
```

## Configuration

### Environment Variables

Set these in your `.env` file or system environment:

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URI` | ✅ Yes | MongoDB connection string |
| `OPENROUTER_API_KEY` | ✅ Yes | OpenRouter API key (FREE tier available) |
| `JWT_SECRET_KEY` | ✅ Yes | Secret key for authentication |
| `GEMINI_API_KEY` | Optional | Google Gemini for embeddings |
| `GROQ_API_KEY` | Optional | Groq as fallback LLM |

### Getting API Keys

1. **MongoDB Atlas** (Free):
   - Go to https://www.mongodb.com/atlas
   - Create free cluster
   - Get connection string

2. **OpenRouter** (Free):
   - Go to https://openrouter.ai/keys
   - Create account
   - Generate API key (free models available!)

3. **JWT Secret**:
   - Generate a secure random string (32+ characters)

## Daily Dose Automation

### Using GitHub Actions (Recommended)

ArionXiv can automatically deliver your daily paper recommendations:

1. Fork the repository
2. Add these secrets in Settings > Secrets:
   - `MONGODB_URI`
   - `OPENROUTER_API_KEY`
   - `JWT_SECRET_KEY`
3. Configure your schedule:
   ```bash
   arionxiv settings daily
   ```

The workflow runs hourly and processes users based on their scheduled time.

### Manual Daily Dose

```bash
# Run daily dose now
arionxiv daily --run

# View today's papers
arionxiv daily --view
```

## Settings & Customization

```bash
# View all settings
arionxiv settings show

# Change theme color
arionxiv settings theme

# Configure API keys
arionxiv settings api

# Set research preferences
arionxiv settings prefs

# Configure daily dose
arionxiv settings daily
```

## Optional Dependencies

```bash
# Advanced PDF processing (OCR, tables)
pip install arionxiv[advanced-pdf]

# ML features (local embeddings)
pip install arionxiv[ml]

# All extras
pip install arionxiv[all]
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `arionxiv search <query>` | Search for papers |
| `arionxiv fetch <paper_id>` | Download a paper |
| `arionxiv analyze <paper_id>` | AI analysis of paper |
| `arionxiv chat [paper_id]` | Chat with papers |
| `arionxiv daily` | Daily dose of papers |
| `arionxiv trending` | Trending topics |
| `arionxiv library` | Manage saved papers |
| `arionxiv settings` | Configure ArionXiv |
| `arionxiv login` | Login to account |
| `arionxiv register` | Create account |
| `arionxiv --help` | Show all commands |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **PyPI**: https://pypi.org/project/arionxiv/
- **GitHub**: https://github.com/ArionDas/ArionXiv
- **Issues**: https://github.com/ArionDas/ArionXiv/issues

---

**Built for researchers, by researchers** 🔬

