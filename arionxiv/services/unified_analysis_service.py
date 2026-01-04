"""
Unified Analysis Service for ArionXiv
Consolidates rag_system.py, rag_chat_system.py, analysis_service.py, analysis_orchestrator.py, and embedding_service.py
Provides comprehensive text analysis, RAG capabilities, chat functionality, orchestration, and embedding services
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import hashlib
import secrets
import logging
from pymongo import IndexModel
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
    np = None
    SentenceTransformer = None

from .unified_database_service import unified_database_service
from .unified_config_service import unified_config_service
from .unified_paper_service import unified_paper_service
from .unified_user_service import unified_user_service
from ..rag_techniques.basic_rag import BasicRAG
from ..prompts import format_prompt

# Import LLM clients from new organized location
from .llm_inference import groq_client, GroqClient
from .llm_inference import OPENROUTER_AVAILABLE
if OPENROUTER_AVAILABLE:
    from .llm_inference import openrouter_client, OpenRouterClient, get_openrouter_client
else:
    openrouter_client = None
    OpenRouterClient = None
    get_openrouter_client = None

# Backward compatibility alias
llm_client = groq_client
LLMClient = GroqClient

load_dotenv()

logger = logging.getLogger(__name__)


class UnifiedAnalysisService:
    """
    Comprehensive analysis service that combines:
    1. RAG (Retrieval-Augmented Generation) system
    2. Paper analysis and processing  
    3. Interactive chat system for papers
    4. Analysis orchestration and workflow management
    5. Embedding services with multiple providers
    """
    
    def __init__(self):
        # Lazy initialization flags
        self._rag = None
        self._rag_initialized = False
        self._openrouter_client = None
        self._openrouter_checked = False
        self._console = None
        self._get_theme_colors = None
        
        # Analysis orchestrator functionality
        self.analysis_orchestrator_enabled = True
        
        logger.info("UnifiedAnalysisService initialized (lazy loading enabled)")
    
    @property
    def analysis_config(self):
        """Lazy load analysis config"""
        return unified_config_service.get_analysis_config()
    
    @property
    def embedding_config(self):
        """Lazy load embedding config"""
        return unified_config_service.get_embedding_config()
    
    @property
    def rag_config(self):
        """Lazy load RAG config"""
        return unified_config_service.get_rag_config()
    
    @property
    def batch_size(self):
        """Get batch size from config"""
        return self.analysis_config["batch_size"]
    
    @property
    def timeout_seconds(self):
        """Get timeout from config"""
        return self.analysis_config["timeout_seconds"]
    
    @property
    def openrouter_client(self):
        """Lazy initialize OpenRouter client"""
        if not self._openrouter_checked:
            self._openrouter_checked = True
            if OPENROUTER_AVAILABLE:
                try:
                    self._openrouter_client = get_openrouter_client()
                    if self._openrouter_client and self._openrouter_client.is_available:
                        logger.info(f"OpenRouter client initialized with model: {self._openrouter_client.get_model_name()}")
                    else:
                        logger.info("OpenRouter client not configured (no API key)")
                        self._openrouter_client = None
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenRouter client: {str(e)}")
                    self._openrouter_client = None
        return self._openrouter_client
    
    @openrouter_client.setter
    def openrouter_client(self, value):
        """Allow setting OpenRouter client"""
        self._openrouter_client = value
        self._openrouter_checked = True
    
    @property
    def rag(self):
        """Lazy initialize RAG system"""
        if not self._rag_initialized:
            self._rag_initialized = True
            self._rag = BasicRAG(
                database_service=unified_database_service,
                config_service=unified_config_service,
                llm_client=llm_client,
                openrouter_client=self.openrouter_client
            )
            logger.info(f"BasicRAG initialized (ML available: {ML_DEPENDENCIES_AVAILABLE})")
        return self._rag
    
    @property
    def console(self):
        """Lazy initialize console"""
        if self._console is None:
            try:
                from cli.ui.theme_system import create_themed_console, get_theme_colors
                self._console = create_themed_console()
                self._get_theme_colors = get_theme_colors
            except ImportError:
                self._console = Console()
                self._get_theme_colors = lambda: {'primary': 'blue', 'secondary': 'cyan'}
        return self._console

    
    # ====================
    # EMBEDDING SERVICE METHODS (delegated to RAG)
    # ====================
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings with automatic fallback"""
        return await self.rag.get_embeddings(texts)
    
    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        return await self.rag.get_single_embedding(text)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.rag.get_embedding_dimension()
    
    def get_embedding_provider_name(self) -> str:
        """Get current provider name"""
        return self.rag.get_embedding_provider_name()
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        return await self.rag.compute_similarity(embedding1, embedding2)
    
    # ====================
    # PAPER ANALYSIS
    # ====================
    
    async def analyze_papers_for_user(self, user_id: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze papers for a specific user"""
        try:
            if not papers:
                logger.warning(f"No papers to analyze for user {user_id}")
                return []
            
            logger.info(f"Starting analysis of {len(papers)} papers for user {user_id}")
            
            # Process papers in batches to avoid overwhelming the LLM service
            analyzed_papers = []
            
            for i in range(0, len(papers), self.batch_size):
                batch = papers[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(papers) + self.batch_size - 1)//self.batch_size}")
                
                batch_results = await self._analyze_batch(user_id, batch)
                analyzed_papers.extend(batch_results)
                
                # Small delay between batches to be respectful to the API
                if i + self.batch_size < len(papers):
                    await asyncio.sleep(1)
            
            logger.info(f"Completed analysis of {len(analyzed_papers)} papers for user {user_id}")
            return analyzed_papers
            
        except Exception as e:
            logger.error(f"Failed to analyze papers for user {user_id}: {e}", exc_info=True)
            return []
    
    async def _analyze_batch(self, user_id: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze a batch of papers"""
        analyzed_papers = []
        
        for paper in papers:
            try:
                analysis_result = await self.analyze_single_paper(user_id, paper)
                if analysis_result:
                    analyzed_papers.append(analysis_result)
            except Exception as e:
                logger.error(f"Failed to analyze paper {paper.get('id', 'unknown')}: {e}", exc_info=True)
                # Continue with other papers even if one fails
                continue
        
        return analyzed_papers
    
    async def analyze_single_paper(self, user_id: str, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single paper using LLM"""
        try:
            paper_id = paper.get('id', paper.get('arxiv_id', 'unknown'))
            logger.info(f"Analyzing paper {paper_id}")
            
            # Prepare paper content for analysis
            content = self._prepare_paper_content(paper)
            
            # Get analysis prompt
            analysis_prompt = format_prompt("paper_analysis", 
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                categories=paper.get("categories", [])
            )
            
            # Call LLM for analysis
            response = await asyncio.wait_for(
                llm_client.get_completion(analysis_prompt),
                timeout=self.timeout_seconds
            )
            
            if not response or response.startswith("Error"):
                logger.error(f"LLM analysis failed for paper {paper_id}")
                return None
            
            # Parse and structure the analysis
            analysis = self._parse_analysis_response(response)
            
            # Create analysis document
            analysis_doc = {
                'paper_id': paper_id,
                'user_id': user_id,
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'abstract': paper.get('abstract', ''),
                'categories': paper.get('categories', []),
                'analysis': analysis,
                'analyzed_at': datetime.utcnow(),
                'analysis_version': '1.0'
            }
            
            # Store analysis in database
            await unified_database_service.insert_one('paper_analyses', analysis_doc)
            
            logger.info(f"Successfully analyzed paper {paper_id}")
            return analysis_doc
            
        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout for paper {paper.get('id', 'unknown')}")
            return None
        except Exception as e:
            logger.error(f"Analysis failed for paper {paper.get('id', 'unknown')}: {e}", exc_info=True)
            return None
    
    def _prepare_paper_content(self, paper: Dict[str, Any]) -> str:
        """Prepare paper content for analysis"""
        content_parts = []
        
        if paper.get('title'):
            content_parts.append(f"Title: {paper['title']}")
        
        if paper.get('authors'):
            authors = ', '.join(paper['authors'])
            content_parts.append(f"Authors: {authors}")
        
        if paper.get('abstract'):
            content_parts.append(f"Abstract: {paper['abstract']}")
        
        if paper.get('categories'):
            categories = ', '.join(paper['categories'])
            content_parts.append(f"Categories: {categories}")
        
        return '\n\n'.join(content_parts)
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response into structured format"""
        try:
            # Try to parse as JSON first
            return json.loads(response)
        except json.JSONDecodeError:
            # If not JSON, create structured response from text
            return {
                'summary': response[:500] + '...' if len(response) > 500 else response,
                'key_points': [],
                'methodology': '',
                'results': '',
                'significance': '',
                'limitations': '',
                'relevance_score': 5  # Default relevance
            }
    
    # ====================
    # RAG SYSTEM
    # ====================
    
    async def add_document_to_index(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to vector index"""
        return await self.rag.add_document_to_index(doc_id, text, metadata)
    
    async def search_similar_documents(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using vector search"""
        return await self.rag.search_similar_documents(query, filters)
    
    # ====================
    # CHAT SYSTEM
    # ====================
    
    async def start_chat_session(self, papers: Union[List[Dict[str, Any]], Dict[str, Any]], user_id: str = "default"):
        """Start interactive chat session with papers"""
        if not isinstance(papers, list):
            papers = [papers]
        
        await self.rag.start_chat_session(papers, user_id)
    
    async def continue_chat_session(self, session: Dict[str, Any], paper_info: Dict[str, Any]):
        """Continue a previous chat session"""
        await self.rag.continue_chat_session(session, paper_info)
    
    async def chat(self, user_name: str, paper_id: str, message: str, session_id: str = None) -> Dict[str, Any]:
        """Process a chat message using the RAG system"""
        return await self.rag._chat_with_session(session_id, message)
    
    # ====================
    # CLEANUP AND MAINTENANCE
    # ====================
    
    async def cleanup_expired_data(self):
        """Clean up expired embeddings and chat sessions"""
        await self.rag.cleanup_expired_data()

    # ========== ANALYSIS ORCHESTRATION METHODS ==========
    
    async def analyze_papers(self, user_id: str, query: Optional[str] = None, 
                           categories: Optional[List[str]] = None,
                           max_papers: int = 10, 
                           analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Orchestrate comprehensive paper analysis workflow.
        
        Args:
            user_id: User ID for the analysis
            query: Optional search query for papers
            categories: Optional ArXiv categories to filter by
            max_papers: Maximum number of papers to analyze
            analysis_type: Type of analysis ('quick', 'comprehensive', 'research')
            
        Returns:
            Dictionary containing analysis results and metrics
        """
        start_time = datetime.utcnow()
        analysis_id = f"analysis_{user_id}_{int(start_time.timestamp())}"
        
        try:
            colors = self._get_theme_colors()
            self.console.print(Panel(
                f"[bold {colors['primary']}]Starting Paper Analysis[/bold {colors['primary']}]\n"
                f"Analysis ID: {analysis_id}\n"
                f"Type: {analysis_type}\n"
                f"Max Papers: {max_papers}",
                title="Analysis Orchestrator",
                border_style=colors['primary']
            ))
            
            # Step 1: Fetch papers based on criteria
            if query:
                papers = await unified_paper_service.search_papers(query, max_results=max_papers)
            else:
                # Get recent papers from specified categories or user preferences
                user_prefs = await unified_user_service.get_user_preferences(user_id)
                target_categories = categories or user_prefs.get('preferred_categories', ['cs.AI', 'cs.LG'])
                
                papers = []
                for category in target_categories:
                    category_papers = await unified_paper_service.fetch_recent_papers(
                        category=category, 
                        max_results=max_papers // len(target_categories)
                    )
                    papers.extend(category_papers)
            
            if not papers:
                return {
                    'analysis_id': analysis_id,
                    'status': 'no_papers',
                    'message': 'No papers found for analysis',
                    'duration': 0
                }
            
            # Step 2: Analyze each paper
            analysis_results = []
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=self.console
            )
            
            with progress:
                task = progress.add_task(f"Analyzing papers...", total=len(papers))
                
                for paper in papers:
                    try:
                        # Generate analysis based on type
                        if analysis_type == "quick":
                            analysis = await self._quick_paper_analysis(paper, user_id)
                        elif analysis_type == "research":
                            analysis = await self._research_paper_analysis(paper, user_id)
                        else:  # comprehensive
                            analysis = await self._comprehensive_paper_analysis(paper, user_id)
                        
                        if analysis:
                            analysis_results.append({
                                'paper_id': paper.get('id'),
                                'title': paper.get('title'),
                                'analysis': analysis,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                        
                        progress.advance(task)
                        
                    except Exception as e:
                        logger.error(f"Error analyzing paper {paper.get('id', 'unknown')}: {e}", exc_info=True)
                        continue
            
            # Step 3: Generate insights and recommendations
            insights = await self._generate_analysis_insights(analysis_results, user_id)
            
            # Step 4: Store orchestration results
            orchestration_doc = {
                'analysis_id': analysis_id,
                'user_id': user_id,
                'type': analysis_type,
                'query': query,
                'categories': categories,
                'papers_analyzed': len(analysis_results),
                'results': analysis_results,
                'insights': insights,
                'start_time': start_time.isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'duration': (datetime.utcnow() - start_time).total_seconds()
            }
            
            await unified_database_service.insert_one('analysis_orchestrations', orchestration_doc)
            
            # Update user statistics
            await self._update_user_analysis_stats(user_id, len(analysis_results))
            
            self.console.print(Panel(
                f"[bold green]Analysis Complete![/bold green]\n"
                f"Papers Analyzed: {len(analysis_results)}\n"
                f"Duration: {orchestration_doc['duration']:.1f}s\n"
                f"Key Insights: {len(insights.get('key_themes', []))}"
            ))
            
            return {
                'analysis_id': analysis_id,
                'status': 'success',
                'papers_analyzed': len(analysis_results),
                'insights': insights,
                'duration': orchestration_doc['duration'],
                'results': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error in analysis orchestration: {e}", exc_info=True)
            return {
                'analysis_id': analysis_id,
                'status': 'error',
                'error': str(e),
                'duration': (datetime.utcnow() - start_time).total_seconds()
            }

    async def run_daily_analysis(self, user_id: str) -> Dict[str, Any]:
        """
        Run daily analysis workflow for a user.
        
        Args:
            user_id: User ID for the daily analysis
            
        Returns:
            Dictionary containing daily analysis results
        """
        try:
            # Get user preferences
            user_prefs = await unified_user_service.get_user_preferences(user_id)
            categories = user_prefs.get('preferred_categories', ['cs.AI', 'cs.LG', 'cs.CV'])
            
            # Fetch yesterday's papers
            yesterday = datetime.utcnow() - timedelta(days=1)
            papers = []
            
            for category in categories:
                category_papers = await unified_paper_service.fetch_papers_by_date(
                    category=category,
                    date=yesterday,
                    max_results=5
                )
                papers.extend(category_papers)
            
            if not papers:
                return {
                    'status': 'no_new_papers',
                    'message': 'No new papers found for yesterday',
                    'date': yesterday.date().isoformat()
                }
            
            # Run analysis
            analysis_result = await self.analyze_papers(
                user_id=user_id,
                max_papers=len(papers),
                analysis_type="quick"
            )
            
            # Generate daily summary
            summary = await self._generate_daily_summary(analysis_result, user_id)
            
            # Store daily analysis
            daily_doc = {
                'user_id': user_id,
                'date': yesterday.date().isoformat(),
                'papers_count': len(papers),
                'analysis_id': analysis_result.get('analysis_id'),
                'summary': summary,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await unified_database_service.insert_one('daily_analyses', daily_doc)
            
            return {
                'status': 'success',
                'date': yesterday.date().isoformat(),
                'papers_analyzed': len(papers),
                'summary': summary,
                'analysis_id': analysis_result.get('analysis_id')
            }
            
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    async def get_weekly_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Generate weekly insights from user's analysis history.
        
        Args:
            user_id: User ID for insights generation
            
        Returns:
            Dictionary containing weekly insights
        """
        try:
            # Get analyses from the past week
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            pipeline = [
                {
                    '$match': {
                        'user_id': user_id,
                        'start_time': {'$gte': week_ago.isoformat()}
                    }
                },
                {
                    '$sort': {'start_time': -1}
                }
            ]
            
            analyses = await unified_database_service.aggregate('analysis_orchestrations', pipeline)
            
            if not analyses:
                return {
                    'status': 'no_data',
                    'message': 'No analyses found in the past week'
                }
            
            # Generate insights
            insights = {
                'period': f"{week_ago.date()} to {datetime.utcnow().date()}",
                'total_analyses': len(analyses),
                'total_papers': sum(a.get('papers_analyzed', 0) for a in analyses),
                'average_papers_per_analysis': sum(a.get('papers_analyzed', 0) for a in analyses) / len(analyses),
                'most_active_day': self._find_most_active_day(analyses),
                'trending_topics': await self._extract_trending_topics(analyses),
                'research_patterns': self._analyze_research_patterns(analyses)
            }
            
            return {
                'status': 'success',
                'insights': insights,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating weekly insights: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    # ========== PRIVATE ORCHESTRATION HELPER METHODS ==========
    
    async def _quick_paper_analysis(self, paper: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
        """Generate quick analysis of a paper."""
        try:
            prompt = format_prompt("quick_analysis",
                title=paper.get("title", ""),
                abstract=paper.get("abstract", "")
            )
            
            analysis_text = await self.llm_client.generate_completion(prompt)
            
            return {
                'type': 'quick',
                'summary': analysis_text[:500],
                'key_points': self._extract_key_points(analysis_text),
                'relevance_score': await unified_user_service.calculate_paper_relevance(paper, user_id)
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}", exc_info=True)
            return None

    async def _comprehensive_paper_analysis(self, paper: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive analysis of a paper."""
        try:
            # This uses the existing analyze_paper method
            return await self.analyze_paper(paper.get('id'), user_id)
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}", exc_info=True)
            return None

    async def _research_paper_analysis(self, paper: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
        """Generate research-focused analysis of a paper."""
        try:
            prompt = format_prompt("research_analysis",
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                categories=paper.get("categories", [])
            )
            
            analysis_text = await self.llm_client.generate_completion(prompt)
            
            # Find related papers using embeddings
            if paper.get('abstract'):
                similar_papers = await self.find_similar_papers(
                    paper.get('abstract'), 
                    limit=3
                )
            else:
                similar_papers = []
            
            return {
                'type': 'research',
                'analysis': analysis_text,
                'methodology_assessment': self._assess_methodology(analysis_text),
                'novelty_score': self._calculate_novelty_score(analysis_text),
                'related_papers': similar_papers,
                'research_implications': self._extract_implications(analysis_text)
            }
            
        except Exception as e:
            logger.error(f"Error in research analysis: {e}", exc_info=True)
            return None

    async def _generate_analysis_insights(self, results: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Generate insights from analysis results."""
        try:
            # Extract themes and patterns
            all_analyses = [r.get('analysis', {}) for r in results]
            
            insights = {
                'key_themes': self._extract_common_themes(all_analyses),
                'methodology_trends': self._analyze_methodology_trends(all_analyses),
                'relevance_distribution': self._analyze_relevance_distribution(all_analyses),
                'recommendation_score': self._calculate_recommendation_score(all_analyses, user_id)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            return {}

    async def _generate_daily_summary(self, analysis_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Generate daily summary from analysis results."""
        try:
            results = analysis_result.get('results', [])
            insights = analysis_result.get('insights', {})
            
            summary = {
                'papers_reviewed': len(results),
                'top_papers': self._get_top_papers(results, limit=3),
                'key_themes': insights.get('key_themes', [])[:5],
                'research_highlights': self._extract_research_highlights(results),
                'personal_recommendations': await self._generate_personal_recommendations(results, user_id)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}", exc_info=True)
            return {}

    async def _update_user_analysis_stats(self, user_id: str, papers_count: int) -> None:
        """Update user's analysis statistics."""
        try:
            stats_update = {
                '$inc': {
                    'total_analyses': 1,
                    'total_papers_analyzed': papers_count
                },
                '$set': {
                    'last_analysis': datetime.utcnow().isoformat()
                }
            }
            
            await unified_database_service.update_one(
                'user_stats',
                {'user_id': user_id},
                stats_update,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating user stats: {e}", exc_info=True)

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from analysis text."""
        # Simple extraction - can be enhanced with NLP
        sentences = text.split('.')
        return [s.strip() for s in sentences if len(s.strip()) > 50][:3]

    def _assess_methodology(self, text: str) -> Dict[str, Any]:
        """Assess methodology from analysis text."""
        # Placeholder for methodology assessment
        return {
            'rigor_score': 0.8,
            'reproducibility': 'high',
            'data_quality': 'good'
        }

    def _calculate_novelty_score(self, text: str) -> float:
        """Calculate novelty score from analysis."""
        # Placeholder for novelty scoring
        return 0.75

    def _extract_implications(self, text: str) -> List[str]:
        """Extract research implications."""
        # Placeholder for implication extraction
        return ["Advances the field", "Practical applications", "Future research directions"]

    def _extract_common_themes(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from analyses."""
        # Placeholder for theme extraction
        return ["Machine Learning", "Deep Learning", "Computer Vision"]

    def _analyze_methodology_trends(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze methodology trends."""
        return {'trending_methods': ['Transformers', 'GANs', 'Reinforcement Learning']}

    def _analyze_relevance_distribution(self, analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze relevance score distribution."""
        return {'high_relevance': 0.3, 'medium_relevance': 0.5, 'low_relevance': 0.2}

    def _calculate_recommendation_score(self, analyses: List[Dict[str, Any]], user_id: str) -> float:
        """Calculate overall recommendation score."""
        return 0.85

    def _get_top_papers(self, results: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        """Get top papers from results."""
        sorted_results = sorted(
            results, 
            key=lambda x: x.get('analysis', {}).get('relevance_score', 0), 
            reverse=True
        )
        return sorted_results[:limit]

    def _extract_research_highlights(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract research highlights."""
        return ["Novel approach to X", "Significant improvement in Y", "New dataset for Z"]

    async def _generate_personal_recommendations(self, results: List[Dict[str, Any]], user_id: str) -> List[str]:
        """Generate personal recommendations."""
        return ["Read papers on topic X", "Explore methodology Y", "Consider collaboration on Z"]

    def _find_most_active_day(self, analyses: List[Dict[str, Any]]) -> str:
        """Find the most active analysis day."""
        day_counts = {}
        for analysis in analyses:
            day = analysis.get('start_time', '').split('T')[0]
            day_counts[day] = day_counts.get(day, 0) + 1
        
        if day_counts:
            return max(day_counts, key=day_counts.get)
        return "No data"

    async def _extract_trending_topics(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Extract trending topics from analyses."""
        # Placeholder for topic extraction
        return ["Large Language Models", "Computer Vision", "Robotics"]

    def _analyze_research_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze research patterns."""
        return {
            'peak_analysis_time': '14:00',
            'average_papers_per_session': 5.2,
            'most_analyzed_categories': ['cs.AI', 'cs.LG']
        }


# Global instance
unified_analysis_service = UnifiedAnalysisService()

# Backwards compatibility
rag_system = unified_analysis_service
rag_chat_system = unified_analysis_service
analysis_service = unified_analysis_service

# Export commonly used functions
analyze_papers_for_user = unified_analysis_service.analyze_papers_for_user
analyze_single_paper = unified_analysis_service.analyze_single_paper
search_similar_documents = unified_analysis_service.search_similar_documents
start_chat_session = unified_analysis_service.start_chat_session
continue_chat_session = unified_analysis_service.continue_chat_session

__all__ = [
    'UnifiedAnalysisService',
    'unified_analysis_service',
    'rag_system',
    'rag_chat_system', 
    'analysis_service',
    'analyze_papers_for_user',
    'analyze_single_paper',
    'search_similar_documents',
    'start_chat_session',
    'continue_chat_session'
]