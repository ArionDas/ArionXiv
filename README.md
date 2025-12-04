# ArionXiv - AI-Powered Research Paper Analysis and Management

**ArionXiv** is a comprehensive Python package for discovering, analyzing, and managing research papers from arXiv with AI-powered insights and organizational features.

## Features

- **Advanced Search**: Intelligent paper discovery with category filtering and relevance scoring
- **AI Analysis**: Deep content analysis using state-of-the-art language models (Groq LLM)
- **Comprehensive Insights**: Daily trend analysis and research direction recommendations
- **Interactive Chat**: Chat with papers to extract specific insights and answers
- **Personal Library**: Organize and manage your research collection
- **Beautiful CLI**: Rich terminal interface with customizable themes
- **Web API**: FastAPI server for integration with web applications
- **Trending Analysis**: Discover trending topics and emerging research areas
- **Secure Authentication**: Local email/password login with JWT session tokens
- **Cloud Database**: MongoDB Atlas integration for scalable data storage

## Installation

### From PyPI (Recommended)

```bash
pip install arionxiv
```

### From Source

```bash
git clone https://github.com/ArionDas/arionxiv.git
cd arionxiv
pip install -e .
```

## Quick Start

### 1. Configuration

Create a `.env` file with your API keys:

```bash
# Copy the template
cp .env.template .env

# Edit with your credentials
MONGODB_URI=your_mongodb_atlas_connection_string
GROQ_API_KEY=your_groq_api_key
```

### 2. CLI Usage

```bash
# Search for papers
arionxiv search "machine learning transformers"

# Fetch and analyze a specific paper
arionxiv fetch 2301.00001
arionxiv analyze 2301.00001

# Daily analysis and trending topics
arionxiv daily
arionxiv trending

# Interactive chat with papers
arionxiv chat

# Manage your library
arionxiv library
```

### 3. Web Server

```bash
# Start the API server
arionxiv-server

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Advanced Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | Required |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM (FREE - primary) | Required |
| `GEMINI_API_KEY` | Google Gemini API key for embeddings | Optional |
| `GROQ_API_KEY` | Groq API key for LLM (fallback) | Optional |
| `DEFAULT_ANALYSIS_MODEL` | LLM model name | `llama-3.3-70b-versatile` |
| `ARXIV_MAX_RESULTS` | Max papers per search | `50` |
| `ANALYSIS_BATCH_SIZE` | Papers per analysis batch | `5` |

### Automated Daily Dose with GitHub Actions

ArionXiv supports automated daily paper analysis via GitHub Actions. This runs hourly and triggers daily dose for users based on their configured schedule time (in UTC).

#### Setup Instructions:

1. **Fork this repository** to your GitHub account

2. **Add repository secrets** in Settings > Secrets and variables > Actions:
   - `MONGODB_URI`: Your MongoDB connection string
   - `OPENROUTER_API_KEY`: Your OpenRouter API key (FREE - get at https://openrouter.ai/keys)
   - `GEMINI_API_KEY`: (Optional) For Gemini embeddings
   - `GROQ_API_KEY`: (Optional) Fallback LLM provider

3. **Enable GitHub Actions** if not already enabled in your fork

4. **Configure your schedule** using the CLI:
   ```bash
   arionxiv settings daily --time 14:00 --enable
   ```
   Note: Times are in UTC. The workflow runs every hour and processes users scheduled for that hour.

#### How It Works:

- The workflow runs at the start of every hour (`:00`)
- It queries the database for users with daily dose enabled and scheduled for the current hour
- Each matching user gets their personalized daily dose generated
- Results are stored in the database and viewable via `arionxiv daily --view`

#### Manual Trigger:

You can manually trigger the workflow from the Actions tab:
1. Go to Actions > "ArionXiv Daily Dose"
2. Click "Run workflow"
3. Optionally specify a `force_hour` (0-23) to test a specific time slot

### Optional Dependencies

Install additional features:

```bash
# Advanced PDF processing
pip install arionxiv[advanced-pdf]

# Machine learning features
pip install arionxiv[ml]

# Enhanced UI components
pip install arionxiv[enhanced-ui]

# Development tools
pip install arionxiv[dev]

# Everything
pip install arionxiv[all]
```

## Database Collections

1. **users** - User accounts and authentication
2. **papers** - Fetched paper metadata and content
3. **analyses** - AI-generated paper analyses
4. **libraries** - User paper collections
5. **preferences** - User settings and preferences
6. **daily_doses** - Daily paper recommendations
7. **chat_sessions** - Interactive chat history
8. **bookmarks** - Saved papers
9. **reading_progress** - Paper reading status
10. **trending** - Trending topics cache
11. **system_metrics** - Performance and usage analytics
12. **cron_logs** - Automated job execution logs

## Setup and Installation

### Prerequisites

- Python 3.8+
- MongoDB (local or cloud instance)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArionDas/ArionXiv.git
   cd ArionXiv
   ``

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ``

3. **Configure MongoDB**
   - Update `config/settings.py` with your MongoDB connection string
   - Ensure MongoDB is running and accessible

4. **Initialize the system**
   ```bash
   cd backend
   python setup_cron.py  # Set up scheduled jobs
   python main.py         # Start the main application
   ``

## Usage

### Starting the System

```bash
# Start the scheduler for automated analysis
python backend/start_scheduler.py

# Run the main application
python backend/main.py
```

### Accessing the UI

Open `frontend/daily-dose-demo.html` in your web browser to see the Daily Dose of Papers interface.

### API Endpoints

The system provides RESTful API endpoints for:
- User authentication and management
- Paper retrieval and searching
- Daily analysis results
- User preferences and bookmarks

## Configuration

Key configuration options in `config/settings.py`:

- **MongoDB Settings**: Database connection and collection names
- **ArXiv API Settings**: Rate limiting and query parameters
- **Analysis Settings**: Scoring algorithms and thresholds
- **Scheduling Settings**: Cron job timing and frequency
- **Authentication**: Local account policies, password hashing, and JWT settings

## Automated Analysis

The system runs four types of scheduled analyses:

1. **Daily Analysis** (9:00 AM): Comprehensive paper analysis and scoring
2. **Hourly Updates** (Every hour): Incremental updates and new paper checks
3. **Weekly Summary** (Sundays): Aggregate analysis and trend reports
4. **Monthly Archive** (1st of month): Data archival and cleanup

## UI Features

The Daily Dose of Papers interface includes:

- **Interactive Paper Cards**: Expandable cards with detailed information
- **Smart Filtering**: Filter by category, relevance, bookmark status
- **Sorting Options**: Sort by relevance, quality, title, or author
- **Bookmark System**: Save papers for later reading
- **Reading Progress**: Track viewed and unread papers
- **PDF Integration**: Direct access to paper PDFs
- **Responsive Design**: Works on desktop and mobile devices

## Analysis Components

### Paper Scoring
- **Relevance Score**: Based on user preferences and research interests
- **Quality Score**: Evaluated using citation patterns, author reputation, and content analysis
- **Overall Score**: Weighted combination of relevance and quality metrics

### AI Summary Generation
- Automated abstract summarization
- Key insight extraction
- Reading time estimation
- Difficulty level assessment

## Security and Authentication

- Local email/password authentication with PBKDF2 hashing
- JWT-based session management
- User data protection
- API rate limiting

## Monitoring and Logging

- Comprehensive logging system
- Performance metrics tracking
- Error monitoring and alerting
- Usage analytics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please create an issue in the repository or contact the development team.

---

**Built for the research community**

