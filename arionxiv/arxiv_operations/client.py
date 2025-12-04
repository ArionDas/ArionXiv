# Arxiv API client for fetching papers
import arxiv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ArxivClient:
    """Client for interacting with Arxiv API"""
    
    def __init__(self):
        self.client = arxiv.Client()
        self.default_page_size = 100
        self.max_results = 100
    
    # Short words to skip in title searches (arXiv doesn't index these well)
    SKIP_WORDS = {'a', 'an', 'the', 'is', 'are', 'be', 'to', 'of', 'in', 'on', 
                  'at', 'by', 'for', 'and', 'or', 'but', 'not', 'all', 'you', 
                  'it', 'its', 'as', 'so', 'if', 'do', 'no', 'up', 'we', 'my'}
    
    def search_papers(self, query: str, max_results: int = None, sort_by=arxiv.SortCriterion.Relevance) -> List[Dict[str, Any]]:
        """Search for papers on Arxiv with relevance scoring"""
        try:
            max_results = max_results or self.default_page_size
            
            # If query already contains arXiv operators (cat:, au:, ti:, abs:, AND, OR), use as-is
            has_operators = any(op in query for op in ['cat:', 'au:', 'ti:', 'abs:', ' AND ', ' OR '])
            
            if has_operators:
                # Query already formatted with operators
                search_query = query
            else:
                # Build title search - skip short common words that arXiv doesn't handle well
                words = [w.strip() for w in query.split() if w.strip()]
                content_words = [w for w in words if w.lower() not in self.SKIP_WORDS]
                
                if content_words:
                    title_parts = [f"ti:{word.title()}" for word in content_words]
                    search_query = " AND ".join(title_parts)
                else:
                    # All words were skipped, use plain query
                    search_query = query
            
            # Fetch more results than requested so we can filter/sort better
            fetch_count = min(max_results * 3, self.max_results) if not has_operators else max_results
            
            search = arxiv.Search(
                query=search_query,
                max_results=min(fetch_count, self.max_results),
                sort_by=sort_by
            )
            
            papers = []
            for result in self.client.results(search):
                paper_data = {
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(author) for author in result.authors],
                    "published": result.published.isoformat() if result.published else None,
                    "updated": result.updated.isoformat() if result.updated else None,
                    "categories": result.categories,
                    "primary_category": result.primary_category,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "doi": result.doi,
                    "journal_ref": result.journal_ref,
                    "comment": result.comment,
                    "links": [{"href": link.href, "title": link.title, "rel": link.rel} for link in result.links]
                }
                papers.append(paper_data)
            
            # Re-score and sort papers by title match quality, then limit to requested count
            if not has_operators and papers:
                papers = self._score_and_sort_papers(papers, query)[:max_results]
            
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return []
    
    def _score_and_sort_papers(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Score papers by how well their title matches the query and sort by score"""
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        scored_papers = []
        for paper in papers:
            title_lower = paper['title'].lower()
            score = 0
            
            # Exact title match (highest priority)
            if title_lower == query_lower:
                score += 1000
            # Title contains the exact query phrase
            elif query_lower in title_lower:
                score += 500
            
            # Count matching words in title
            title_words = set(title_lower.split())
            matching_words = query_words & title_words
            score += len(matching_words) * 50
            
            # Bonus for shorter titles (more likely to be exact match)
            if len(title_words) <= len(query_words) + 2:
                score += 100
            
            # Bonus for title starting with query words
            if title_lower.startswith(query_lower.split()[0]):
                score += 75
            
            scored_papers.append((score, paper))
        
        # Sort by score descending
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        return [paper for score, paper in scored_papers]
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper by its Arxiv ID"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            
            for result in self.client.results(search):
                paper_data = {
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(author) for author in result.authors],
                    "published": result.published.isoformat() if result.published else None,
                    "updated": result.updated.isoformat() if result.updated else None,
                    "categories": result.categories,
                    "primary_category": result.primary_category,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "doi": result.doi,
                    "journal_ref": result.journal_ref,
                    "comment": result.comment,
                    "links": [{"href": link.href, "title": link.title, "rel": link.rel} for link in result.links]
                }
                return paper_data
            
            return None
        except Exception as e:
            logger.error(f"Error fetching paper {arxiv_id}: {str(e)}")
            return None
    
    def get_recent_papers(self, category: str = None, days: int = 7, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get recent papers from the last N days"""
        try:
            # Build query for recent papers
            query_parts = []
            
            if category:
                query_parts.append(f"cat:{category}")
            
            # Date filter (Arxiv doesn't support date ranges directly, so we'll filter results)
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = " AND ".join(query_parts) if query_parts else "all:machine learning"
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in self.client.results(search):
                # Filter by date
                if result.published and result.published.replace(tzinfo=None) >= cutoff_date:
                    paper_data = {
                        "arxiv_id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "abstract": result.summary,
                        "authors": [str(author) for author in result.authors],
                        "published": result.published.isoformat() if result.published else None,
                        "updated": result.updated.isoformat() if result.updated else None,
                        "categories": result.categories,
                        "primary_category": result.primary_category,
                        "pdf_url": result.pdf_url,
                        "entry_id": result.entry_id,
                        "doi": result.doi,
                        "journal_ref": result.journal_ref,
                        "comment": result.comment
                    }
                    papers.append(paper_data)
            
            logger.info(f"Found {len(papers)} recent papers in category: {category}")
            return papers
        except Exception as e:
            logger.error(f"Error fetching recent papers: {str(e)}")
            return []
    
    def get_papers_by_category(self, category: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get papers by category"""
        try:
            query = f"cat:{category}"
            return self.search_papers(query, max_results)
        except Exception as e:
            logger.error(f"Error fetching papers by category {category}: {str(e)}")
            return []
    
    def get_papers_by_author(self, author: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get papers by author"""
        try:
            query = f"au:{author}"
            return self.search_papers(query, max_results)
        except Exception as e:
            logger.error(f"Error fetching papers by author {author}: {str(e)}")
            return []
    
    def get_trending_papers(self, category: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """Get trending papers (most recent with high engagement indicators)"""
        try:
            # For now, we'll use recent papers as a proxy for trending
            # In a full implementation, this could consider download counts, citations, etc.
            return self.get_recent_papers(category=category, days=days, max_results=30)
        except Exception as e:
            logger.error(f"Error fetching trending papers: {str(e)}")
            return []

# Global instance
arxiv_client = ArxivClient()
