<p align="center">
  <img
    src="https://github.com/user-attachments/assets/cf79dfe7-a4c0-4a9c-8c2f-c0b663049457"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

> [!NOTE]
Let's face it, nobody has the bandwidth, nor the patience to go through entire papers every single day. <br>
But that doesn't mean you will have to miss out on all these interesting launches everyday! <br>
Use [ArionXiv](https://pypi.org/project/arionxiv/), automate it! Its absolutely **FREE** for all!

---

## Installation

```bash
pip install arionxiv
```
> [!IMPORTANT]
> If the command is not found after installation, add Python scripts to PATH:

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

### Welcome to ArionXiv
> [!IMPORTANT]
> The first command you should always run is this, it'll display the welcome screen and guide you through the initial setup.:

```bash
arionxiv welcome
```

### Create an Account

> [!IMPORTANT]
> First-time users must create an account to use ArionXiv:

```bash
arionxiv register   # Create a new account (required for first-time users)
arionxiv login      # Login to existing account
```

Once registered, you can access all features including paper search, AI analysis, chat, and personalized recommendations.

That's it. No API keys or configuration required.

---

## Features

### 1. Daily Dose of AI

Configure your keywords, time, and number of papers (max 10) - and you'll have analysis of all these papers ready for you the next time you come in. <br>
(I use it to run analysis overnight, so I can quickly skim through the papers in the morning.)

```bash
arionxiv daily
arionxiv daily --run
arionxiv daily --dose
```

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/bf6bde86-c0d9-4a33-bdba-20d97014affc"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

Configure schedule and preferences:

```bash
arionxiv settings daily
```

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/36bb9dd4-e1ce-4720-a32b-c61a0a608952"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

---
### 2. Paper Analysis

Get exhaustive analysis on the paper.

```bash
arionxiv search "transformer architecture"
# Select a paper → Choose "Analyze"
```

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/a8979b3c-e6ac-4411-a77a-31759c8fe399"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

---

### 3. Chat with Papers

Interactive RAG-based Q&A with any paper. Supports session persistence and history.

```bash
arionxiv chat
```

**Features:**
- Context-aware responses using paper content
- Chat history maintained indefinitely
- Chat session persisted for 24 hours for reuse
- Chat history also persisted for reuse

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/6d6f9c44-2b0d-49b0-acbf-8e4677e538ac"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

---

### 4. Personal Library

Save papers and manage your research collection.

```bash
arionxiv library
arionxiv settings papers
```

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/002cec70-6c96-4900-af4d-2e3057800cc4"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

---

### 6. Trending Papers

Discover trending research topics and papers.

```bash
arionxiv trending
```
> [!NOTE]
> Feature coming soon...!

---

### 7. Themes

Customizable terminal interface with multiple color themes.

```bash
arionxiv settings theme
```

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/38428099-c35e-407b-ad41-6cec36cd3c84"
    alt="image"
    style="max-width:100%; height:auto;"
  />
</p>

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
arionxiv settings api          # Configure optional API keys (OpenRouter, Gemini, Groq)
arionxiv settings preferences  # Research preferences
arionxiv settings daily        # Daily dose schedule
arionxiv settings papers       # Manage saved papers
```

### Self-Hosting 
> [!NOTE]
> **Note:** Regular users do NOT need to self-host. ArionXiv automatically connects to our hosted backend service. This section is only for people who want to run models through their own APIs.

Using command "arionxiv settings api":

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM |
| `GEMINI_API_KEY` | Google Gemini embeddings (optional) |
| `GROQ_API_KEY` | Fallback LLM provider (optional) |

---

> [!IMPORTANT]
> Don't get freaked out by the python libraries getting installed! It's completely normal. 

> [!WARNING]
> The application is built in a way that it will work perfectly given all the packages have been installed correctly. In case you observe any inconsistencies/warnings, it could either be some packages didn't get installed correctly, or there might have been an outdated version of the same existing in your system.</span>

---

## Links

- PyPI: https://pypi.org/project/arionxiv/
- GitHub: https://github.com/ArionDas/ArionXiv
- Issues: https://github.com/ArionDas/ArionXiv/issues

---

## License

MIT License
