"""
Unified Daily Dose Service for ArionXiv
Handles the complete daily dose workflow:
- Fetching papers based on user keywords via arXiv search
- Extracting text and generating embeddings
- Creating thorough analysis for each paper
- Storing results in MongoDB against user_id
- Replacing previous day's analysis when cron runs again
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time

from .unified_database_service import unified_database_service
from .unified_config_service import unified_config_service
from .unified_pdf_service import unified_pdf_processor
from ..arxiv_operations.client import arxiv_client
from ..prompts import format_prompt

# Use OpenRouter as primary LLM provider (FREE tier available)
# Falls back to Groq if OpenRouter unavailable
def _get_llm_client():
    """Get the appropriate LLM client based on environment."""
    provider = os.getenv("RAG_LLM_PROVIDER", "openrouter").lower()
    
    if provider == "openrouter" or os.getenv("OPENROUTER_API_KEY"):
        from .llm_inference.openrouter_client import openrouter_client
        return openrouter_client
    else:
        from .llm_client import llm_client
        return llm_client

logger = logging.getLogger(__name__)


# Rate limiting constants
ARXIV_REQUEST_DELAY = 3.0  # seconds between arXiv requests
LLM_REQUEST_DELAY = 1.0    # seconds between LLM requests
MAX_PAPERS_V1 = 10         # Maximum papers for version 1


class UnifiedDailyDoseService:
    """
    Service for managing daily dose paper recommendations and analysis.
    
    Features:
    - Fetches papers based on user-saved keywords from DB
    - Respects rate limits for arXiv and LLM APIs
    - Generates embeddings and thorough analysis for each paper
    - Stores everything in MongoDB against user_id
    - Replaces previous day's analysis on each cron run
    """
    
    def __init__(self):
        self.max_papers = MAX_PAPERS_V1
        self.arxiv_delay = ARXIV_REQUEST_DELAY
        self.llm_delay = LLM_REQUEST_DELAY
        logger.info("UnifiedDailyDoseService initialized")
    
    async def get_user_daily_dose_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's daily dose settings from the database or API.
        
        Returns settings including:
        - keywords: List of search keywords
        - max_papers: Number of papers to fetch (max 10)
        - scheduled_time: Time for cron job (HH:MM format)
        - enabled: Whether daily dose is enabled
        """
        try:
            # Try API first for hosted users (no local MongoDB)
            try:
                from ..cli.utils.api_client import api_client
                if api_client.is_authenticated():
                    result = await api_client.get_settings()
                    if result.get("success"):
                        settings = result.get("settings", {})
                        daily_dose = settings.get("daily_dose", {})
                        preferences = settings.get("preferences", {})
                        return {
                            "success": True,
                            "settings": {
                                "keywords": daily_dose.get("keywords", preferences.get("keywords", [])),
                                "max_papers": min(daily_dose.get("max_papers", 5), self.max_papers),
                                "scheduled_time": daily_dose.get("scheduled_time", None),
                                "enabled": daily_dose.get("enabled", False),
                                "categories": preferences.get("categories", ["cs.AI", "cs.LG"])
                            }
                        }
            except Exception as api_err:
                logger.debug(f"API settings fetch failed, trying local DB: {api_err}")
            
            # Fall back to local MongoDB
            if unified_database_service.db is None:
                try:
                    await unified_database_service.connect_mongodb()
                except Exception as db_err:
                    logger.debug(f"Local MongoDB not available: {db_err}")
                    return {
                        "success": False,
                        "message": "No database connection available. Please ensure you're logged in.",
                        "settings": self._get_default_settings()
                    }
            
            # Get user preferences from local DB
            user = await unified_database_service.find_one("users", {"_id": user_id})
            if not user:
                # Try alternate lookup
                from bson import ObjectId
                try:
                    user = await unified_database_service.find_one("users", {"_id": ObjectId(user_id)})
                except Exception:
                    pass
            
            if not user:
                return {
                    "success": False,
                    "message": "User not found",
                    "settings": self._get_default_settings()
                }
            
            preferences = user.get("preferences", {})
            daily_dose_settings = preferences.get("daily_dose", {})
            
            return {
                "success": True,
                "settings": {
                    "keywords": daily_dose_settings.get("keywords", preferences.get("keywords", [])),
                    "max_papers": min(daily_dose_settings.get("max_papers", 5), self.max_papers),
                    "scheduled_time": daily_dose_settings.get("scheduled_time", None),
                    "enabled": daily_dose_settings.get("enabled", False),
                    "categories": preferences.get("categories", ["cs.AI", "cs.LG"])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily dose settings for user {user_id}: {e}")
            return {
                "success": False,
                "message": str(e),
                "settings": self._get_default_settings()
            }
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Return default daily dose settings."""
        return {
            "keywords": [],
            "max_papers": 5,
            "scheduled_time": None,
            "enabled": False,
            "categories": ["cs.AI", "cs.LG"]
        }
    
    async def update_user_daily_dose_settings(
        self, 
        user_id: str, 
        keywords: List[str] = None,
        max_papers: int = None,
        scheduled_time: str = None,
        enabled: bool = None
    ) -> Dict[str, Any]:
        """
        Update user's daily dose settings in the database.
        """
        try:
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            # Build update dict
            updates = {}
            if keywords is not None:
                updates["preferences.daily_dose.keywords"] = keywords
                updates["preferences.keywords"] = keywords  # Also update main keywords
            if max_papers is not None:
                updates["preferences.daily_dose.max_papers"] = min(max_papers, self.max_papers)
            if scheduled_time is not None:
                updates["preferences.daily_dose.scheduled_time"] = scheduled_time
            if enabled is not None:
                updates["preferences.daily_dose.enabled"] = enabled
            
            updates["preferences.daily_dose.updated_at"] = datetime.utcnow()
            
            # Update user document
            from bson import ObjectId
            try:
                filter_query = {"_id": ObjectId(user_id)}
            except Exception:
                filter_query = {"_id": user_id}
            
            result = await unified_database_service.update_one(
                "users",
                filter_query,
                {"$set": updates}
            )
            
            if result:
                logger.info(f"Updated daily dose settings for user {user_id}")
                return {"success": True, "message": "Settings updated successfully"}
            else:
                return {"success": False, "message": "Failed to update settings"}
                
        except Exception as e:
            logger.error(f"Failed to update daily dose settings: {e}")
            return {"success": False, "message": str(e)}
    
    async def execute_daily_dose(self, user_id: str, progress_callback=None) -> Dict[str, Any]:
        """
        Execute the daily dose workflow for a user.
        
        Steps:
        1. Get user's keywords from DB
        2. Search arXiv for matching papers (with rate limiting)
        3. Extract text from each paper
        4. Generate embeddings for each paper
        5. Create thorough analysis for each paper
        6. Store everything in DB, replacing previous day's data
        
        Args:
            user_id: User ID to run daily dose for
            progress_callback: Optional callback function(step: str, detail: str) for progress updates
        
        Returns:
            Dict with success status, papers processed, and analysis_id
        """
        def log_progress(step: str, detail: str = ""):
            """Log progress both to logger and callback"""
            logger.info(f"{step}: {detail}" if detail else step)
            if progress_callback:
                progress_callback(step, detail)
        
        start_time = datetime.utcnow()
        log_progress("Starting daily dose", f"User: {user_id}")
        
        try:
            # Step 1: Get user settings
            log_progress("Loading settings", "Fetching user preferences...")
            settings_result = await self.get_user_daily_dose_settings(user_id)
            if not settings_result["success"]:
                return {
                    "success": False,
                    "message": f"Failed to get user settings: {settings_result['message']}",
                    "papers_count": 0
                }
            
            settings = settings_result["settings"]
            keywords = settings["keywords"]
            categories = settings["categories"]
            max_papers = settings["max_papers"]
            
            if not keywords and not categories:
                return {
                    "success": False,
                    "message": "No keywords or categories configured. Please set up your preferences in settings.",
                    "papers_count": 0
                }
            
            log_progress("Settings loaded", f"Keywords: {len(keywords)}, Categories: {len(categories)}, Max papers: {max_papers}")
            
            # Step 2: Search arXiv for papers
            log_progress("Searching arXiv", "Finding papers matching your keywords...")
            papers = await self._fetch_papers_from_arxiv(keywords, categories, max_papers)
            
            if not papers:
                return {
                    "success": True,
                    "message": "No new papers found matching your keywords.",
                    "papers_count": 0
                }
            
            log_progress("Papers found", f"Found {len(papers)} papers matching your criteria")
            
            # Step 3-5: Process each paper (text extraction, embeddings, analysis)
            processed_papers = []
            for i, paper in enumerate(papers):
                paper_title = paper.get('title', 'Unknown')[:50]
                log_progress(f"Analyzing paper {i+1}/{len(papers)}", paper_title)
                
                try:
                    processed_paper = await self._process_paper(paper, user_id)
                    processed_papers.append(processed_paper)
                    log_progress(f"Paper {i+1} analyzed", f"Score: {processed_paper.get('analysis', {}).get('relevance_score', 'N/A')}/10")
                except Exception as e:
                    logger.error(f"Failed to process paper {paper.get('arxiv_id')}: {e}")
                    # Continue with other papers
                    processed_papers.append({
                        "paper": paper,
                        "analysis": None,
                        "error": str(e)
                    })
                    log_progress(f"Paper {i+1} failed", str(e)[:50])
                
                # Rate limiting between papers
                if i < len(papers) - 1:
                    await asyncio.sleep(self.llm_delay)
            
            # Step 6: Store in DB, replacing previous day's data
            log_progress("Saving to database", "Storing analysis results...")
            analysis_result = await self._store_daily_analysis(user_id, processed_papers, start_time)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            log_progress("Complete", f"Saved {len(processed_papers)} papers in {execution_time:.1f}s")
            
            # Build dose object for return
            successful_papers = [p for p in processed_papers if p.get("analysis") and not p.get("error")]
            
            # Calculate summary statistics
            avg_relevance = 0
            if successful_papers:
                scores = [p["analysis"].get("relevance_score", 5) for p in successful_papers]
                avg_relevance = sum(scores) / len(scores)
            
            dose = {
                "papers": [
                    {
                        "arxiv_id": p["paper"]["arxiv_id"],
                        "title": p["paper"]["title"],
                        "authors": p["paper"]["authors"],
                        "abstract": p["paper"]["abstract"],
                        "categories": p["paper"]["categories"],
                        "published": p["paper"]["published"],
                        "pdf_url": p["paper"]["pdf_url"],
                        "analysis": p["analysis"],
                        "relevance_score": p["analysis"].get("relevance_score", 5) if p["analysis"] else 0
                    }
                    for p in processed_papers if p.get("analysis")
                ],
                "summary": {
                    "total_papers": len(processed_papers),
                    "successful_analyses": len(successful_papers),
                    "avg_relevance_score": round(avg_relevance, 2),
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return {
                "success": True,
                "message": "Daily dose generated successfully",
                "papers_count": len(processed_papers),
                "analysis_id": analysis_result.get("analysis_id"),
                "execution_time": execution_time,
                "dose": dose
            }
            
        except Exception as e:
            logger.error(f"Daily dose execution failed for user {user_id}: {e}")
            return {
                "success": False,
                "message": f"Daily dose execution failed: {str(e)}",
                "papers_count": 0
            }
    
    async def _fetch_papers_from_arxiv(
        self, 
        keywords: List[str], 
        categories: List[str],
        max_papers: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from arXiv based on keywords and categories.
        Uses Atlas-style search combining keywords with category filters.
        """
        try:
            # Normalize keywords - split any space-separated strings into individual keywords
            normalized_keywords = []
            for kw in keywords:
                if isinstance(kw, str):
                    # Split by common separators and filter empty strings
                    # Handle both comma-separated and space-separated keywords
                    if ',' in kw:
                        # Comma-separated: "DPO, alignment, RL"
                        parts = [p.strip() for p in kw.split(',') if p.strip()]
                    else:
                        # Space-separated single words or phrases
                        # Keep multi-word phrases together (e.g., "test time scaling")
                        parts = [p.strip() for p in kw.split() if p.strip()]
                    normalized_keywords.extend(parts)
                else:
                    normalized_keywords.append(str(kw))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for kw in normalized_keywords:
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    unique_keywords.append(kw)
            
            logger.info(f"Normalized keywords: {unique_keywords}")
            
            # Build search query
            query_parts = []
            
            # Add category filter
            if categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                query_parts.append(f"({cat_query})")
            
            # Add keyword filter - don't use quotes for single words
            if unique_keywords:
                # For single words, just use the word; for phrases, use quotes
                kw_parts = []
                for kw in unique_keywords:
                    if ' ' in kw:
                        kw_parts.append(f'"{kw}"')  # Phrase
                    else:
                        kw_parts.append(kw)  # Single word
                kw_query = " OR ".join(kw_parts)
                query_parts.append(f"({kw_query})")
            
            # Combine with AND
            if query_parts:
                full_query = " AND ".join(query_parts)
            else:
                # Fallback to general CS papers
                full_query = "cat:cs.AI OR cat:cs.LG"
            
            logger.info(f"Searching arXiv with query: {full_query}")
            
            # Rate limit before API call
            await asyncio.sleep(self.arxiv_delay)
            
            # Search arXiv - sort by SubmittedDate (descending) to get most recent papers
            import arxiv
            papers = arxiv_client.search_papers(
                query=full_query,
                max_results=min(max_papers * 2, 20),  # Fetch extra to allow filtering
                sort_by=arxiv.SortCriterion.SubmittedDate  # Most recent first!
            )
            
            # Filter to most recent papers and limit
            papers = papers[:max_papers]
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to fetch papers from arXiv: {e}")
            return []
    
    async def _process_paper(self, paper: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Process a single paper: extract text, generate embeddings, create analysis.
        """
        arxiv_id = paper.get("arxiv_id", "unknown")
        
        # Use abstract as primary text source (faster than PDF extraction)
        paper_text = f"Title: {paper.get('title', '')}\n\n"
        paper_text += f"Authors: {', '.join(paper.get('authors', []))}\n\n"
        paper_text += f"Abstract: {paper.get('abstract', '')}\n\n"
        paper_text += f"Categories: {', '.join(paper.get('categories', []))}"
        
        # Generate thorough analysis using LLM
        analysis = await self._generate_paper_analysis(paper)
        
        # Generate embeddings for the paper
        embeddings = await self._generate_embeddings(paper_text)
        
        return {
            "paper": {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "categories": paper.get("categories", []),
                "published": paper.get("published"),
                "pdf_url": paper.get("pdf_url"),
                "entry_id": paper.get("entry_id")
            },
            "text_content": paper_text,
            "embeddings": embeddings,
            "analysis": analysis,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _generate_paper_analysis(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate thorough analysis for a paper using LLM.
        """
        try:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            categories = paper.get("categories", [])
            authors = paper.get("authors", [])
            
            # Use centralized prompt from prompts module
            prompt = format_prompt(
                "daily_dose_analysis",
                title=title,
                authors=', '.join(authors[:5]) + ('...' if len(authors) > 5 else ''),
                categories=', '.join(categories),
                abstract=abstract
            )

            # Rate limit before LLM call
            await asyncio.sleep(self.llm_delay)
            
            # Get LLM response using the appropriate client
            client = _get_llm_client()
            response = await client.get_completion(prompt)
            
            if not response or response.startswith("Error"):
                logger.warning(f"LLM analysis failed for paper: {paper.get('arxiv_id')}")
                return self._get_fallback_analysis(paper)
            
            # Parse the response into structured format
            analysis = self._parse_analysis_response(response)
            analysis["raw_response"] = response
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate analysis for paper: {e}")
            return self._get_fallback_analysis(paper)
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis."""
        sections = {
            "summary": "",
            "key_findings": [],
            "methodology": "",
            "significance": "",
            "limitations": "",
            "relevance_score": 5
        }
        
        try:
            lines = response.split("\n")
            current_section = None
            current_content = []
            
            for line in lines:
                line_lower = line.lower().strip()
                
                if "summary" in line_lower and ":" in line:
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = "summary"
                    current_content = [line.split(":", 1)[-1].strip()]
                elif "key finding" in line_lower or "findings" in line_lower:
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = "key_findings"
                    current_content = []
                elif "methodology" in line_lower and ":" in line:
                    if current_section:
                        if current_section == "key_findings":
                            sections[current_section] = current_content
                        else:
                            sections[current_section] = "\n".join(current_content).strip()
                    current_section = "methodology"
                    current_content = [line.split(":", 1)[-1].strip()]
                elif "significance" in line_lower and ":" in line:
                    if current_section:
                        if current_section == "key_findings":
                            sections[current_section] = current_content
                        else:
                            sections[current_section] = "\n".join(current_content).strip()
                    current_section = "significance"
                    current_content = [line.split(":", 1)[-1].strip()]
                elif "limitation" in line_lower and ":" in line:
                    if current_section:
                        if current_section == "key_findings":
                            sections[current_section] = current_content
                        else:
                            sections[current_section] = "\n".join(current_content).strip()
                    current_section = "limitations"
                    current_content = [line.split(":", 1)[-1].strip()]
                elif "relevance" in line_lower and "score" in line_lower:
                    if current_section:
                        if current_section == "key_findings":
                            sections[current_section] = current_content
                        else:
                            sections[current_section] = "\n".join(current_content).strip()
                    # Try to extract score
                    import re
                    score_match = re.search(r'(\d+)', line)
                    if score_match:
                        sections["relevance_score"] = min(10, max(1, int(score_match.group(1))))
                    current_section = None
                    current_content = []
                elif current_section:
                    # Handle bullet points for key findings
                    if current_section == "key_findings":
                        cleaned = line.strip().lstrip("-*").strip()
                        if cleaned:
                            current_content.append(cleaned)
                    else:
                        current_content.append(line)
            
            # Save last section
            if current_section:
                if current_section == "key_findings":
                    sections[current_section] = current_content
                else:
                    sections[current_section] = "\n".join(current_content).strip()
            
            # If parsing failed, use raw response as summary
            if not sections["summary"]:
                sections["summary"] = response[:500] + "..." if len(response) > 500 else response
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            sections["summary"] = response[:500] + "..." if len(response) > 500 else response
        
        return sections
    
    def _get_fallback_analysis(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Return a fallback analysis when LLM fails."""
        abstract = paper.get("abstract", "")
        return {
            "summary": abstract[:300] + "..." if len(abstract) > 300 else abstract,
            "key_findings": ["Analysis unavailable - see abstract for details"],
            "methodology": "See paper for methodology details",
            "significance": "Please review the paper for significance",
            "limitations": "Unable to determine",
            "relevance_score": 5,
            "error": "LLM analysis failed"
        }
    
    async def _generate_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for the paper text."""
        try:
            from .unified_analysis_service import unified_analysis_service
            embedding = await unified_analysis_service.get_single_embedding(text[:4000])  # Limit text length
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            return None
    
    async def _store_daily_analysis(
        self, 
        user_id: str, 
        processed_papers: List[Dict[str, Any]],
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        Store daily analysis in MongoDB, replacing any existing analysis for today.
        """
        try:
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            # Get today's date boundaries
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            # Delete any existing daily dose for this user (replace logic)
            await unified_database_service.db.daily_dose.delete_many({
                "user_id": user_id
            })
            
            # Prepare the daily dose document
            successful_papers = [p for p in processed_papers if p.get("analysis") and not p.get("error")]
            
            # Calculate summary statistics
            avg_relevance = 0
            if successful_papers:
                scores = [p["analysis"].get("relevance_score", 5) for p in successful_papers]
                avg_relevance = sum(scores) / len(scores)
            
            # Get all categories covered
            categories_covered = set()
            for p in successful_papers:
                categories_covered.update(p["paper"].get("categories", []))
            
            # Get top keywords from papers
            all_text = " ".join([p["paper"].get("title", "") for p in successful_papers])
            top_keywords = self._extract_top_keywords(all_text)
            
            daily_dose_doc = {
                "user_id": user_id,
                "generated_at": datetime.utcnow(),
                "date": today.isoformat(),
                "papers": [
                    {
                        "arxiv_id": p["paper"]["arxiv_id"],
                        "title": p["paper"]["title"],
                        "authors": p["paper"]["authors"],
                        "abstract": p["paper"]["abstract"],
                        "categories": p["paper"]["categories"],
                        "published": p["paper"]["published"],
                        "pdf_url": p["paper"]["pdf_url"],
                        "analysis": p["analysis"],
                        "relevance_score": p["analysis"].get("relevance_score", 5) if p["analysis"] else 0
                    }
                    for p in processed_papers
                ],
                "summary": {
                    "total_papers": len(processed_papers),
                    "successful_analyses": len(successful_papers),
                    "avg_relevance_score": round(avg_relevance, 2),
                    "categories_covered": list(categories_covered),
                    "top_keywords": top_keywords
                },
                "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "created_at": datetime.utcnow()
            }
            
            # Insert new daily dose
            result = await unified_database_service.db.daily_dose.insert_one(daily_dose_doc)
            
            logger.info(f"Stored daily dose for user {user_id}, analysis_id: {result.inserted_id}")
            
            return {
                "success": True,
                "analysis_id": str(result.inserted_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to store daily analysis: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_top_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract top keywords from text (simple frequency-based)."""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common words
        stopwords = {
            "with", "from", "this", "that", "have", "been", "were", "their",
            "which", "when", "where", "what", "will", "would", "could", "should",
            "using", "based", "approach", "method", "paper", "model", "learning",
            "neural", "network", "data", "results", "show", "propose", "proposed"
        }
        
        word_counts = {}
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]
    
    async def get_user_daily_dose(self, user_id: str) -> Dict[str, Any]:
        """
        Get the latest daily dose for a user from the database.
        """
        try:
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            # Get the most recent daily dose
            daily_dose = await unified_database_service.db.daily_dose.find_one(
                {"user_id": user_id},
                sort=[("generated_at", -1)]
            )
            
            if not daily_dose:
                return {
                    "success": False,
                    "message": "No daily dose found. Generate one using 'arionxiv daily --run'",
                    "data": None
                }
            
            # Convert ObjectId to string
            daily_dose["_id"] = str(daily_dose["_id"])
            
            return {
                "success": True,
                "data": daily_dose
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily dose for user {user_id}: {e}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }
    
    async def get_paper_analysis(self, user_id: str, arxiv_id: str) -> Dict[str, Any]:
        """
        Get the stored analysis for a specific paper from the daily dose.
        """
        try:
            daily_dose_result = await self.get_user_daily_dose(user_id)
            
            if not daily_dose_result["success"]:
                return daily_dose_result
            
            papers = daily_dose_result["data"].get("papers", [])
            
            for paper in papers:
                if paper.get("arxiv_id") == arxiv_id:
                    return {
                        "success": True,
                        "paper": paper,
                        "analysis": paper.get("analysis")
                    }
            
            return {
                "success": False,
                "message": f"Paper {arxiv_id} not found in daily dose"
            }
            
        except Exception as e:
            logger.error(f"Failed to get paper analysis: {e}")
            return {
                "success": False,
                "message": str(e)
            }


# Global instance
unified_daily_dose_service = UnifiedDailyDoseService()

# Backwards compatibility
daily_dose_service = unified_daily_dose_service

__all__ = [
    'UnifiedDailyDoseService',
    'unified_daily_dose_service',
    'daily_dose_service'
]
