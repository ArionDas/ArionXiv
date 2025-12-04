# Utility functions for Arxiv operations
import re
import hashlib
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ArxivUtils:
    """Utility functions for Arxiv operations"""
    
    @staticmethod
    def normalize_arxiv_id(arxiv_id: str) -> str:
        """
        Normalize Arxiv ID by removing version numbers.
        This ensures consistent ID format across the application.
        Examples:
            '2502.03095v1' -> '2502.03095'
            '2502.03095v7' -> '2502.03095'
            '2502.03095' -> '2502.03095'
        """
        if not arxiv_id:
            return arxiv_id
        # Remove version suffix (v1, v2, ..., v99)
        return re.sub(r'v\d+$', '', arxiv_id.strip())
    
    @staticmethod
    def clean_arxiv_id(arxiv_id: str) -> str:
        """Clean and normalize Arxiv ID"""
        try:
            # Remove common prefixes and clean
            cleaned = arxiv_id.replace("http://arxiv.org/abs/", "")
            cleaned = cleaned.replace("https://arxiv.org/abs/", "")
            cleaned = cleaned.replace("arxiv:", "")
            
            # Extract just the ID part
            if "/" in cleaned:
                cleaned = cleaned.split("/")[-1]
            
            return cleaned.strip()
        except Exception as e:
            logger.error(f"Error cleaning arxiv ID {arxiv_id}: {str(e)}")
            return arxiv_id
    
    @staticmethod
    def extract_arxiv_id_from_url(url: str) -> Optional[str]:
        """Extract Arxiv ID from various URL formats"""
        try:
            # Common patterns for Arxiv URLs
            patterns = [
                r"arxiv\.org/abs/([^/?]+)",
                r"arxiv\.org/pdf/([^/?]+)",
                r"arxiv:([^/?]+)",
                r"/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
                r"/([a-z-]+/[0-9]{7}(?:v[0-9]+)?)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url, re.IGNORECASE)
                if match:
                    return ArxivUtils.clean_arxiv_id(match.group(1))
            
            return None
        except Exception as e:
            logger.error(f"Error extracting arxiv ID from URL {url}: {str(e)}")
            return None
    
    @staticmethod
    def validate_arxiv_id(arxiv_id: str) -> bool:
        """Validate if a string is a valid Arxiv ID"""
        try:
            cleaned_id = ArxivUtils.clean_arxiv_id(arxiv_id)
            
            # New format: YYMM.NNNN[vN]
            new_format = re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?$", cleaned_id)
            
            # Old format: subject-class/YYMMnnn[vN]
            old_format = re.match(r"^[a-z-]+/[0-9]{7}(?:v[0-9]+)?$", cleaned_id)
            
            return bool(new_format or old_format)
        except Exception as e:
            logger.error(f"Error validating arxiv ID {arxiv_id}: {str(e)}")
            return False
    
    @staticmethod
    def generate_paper_hash(paper_data: Dict[str, Any]) -> str:
        """Generate a unique hash for a paper"""
        try:
            # Use arxiv_id, title, and first author for hash
            hash_string = ""
            hash_string += paper_data.get("arxiv_id", "")
            hash_string += paper_data.get("title", "")
            
            authors = paper_data.get("authors", [])
            if authors:
                hash_string += authors[0]
            
            return hashlib.md5(hash_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating paper hash: {str(e)}")
            return hashlib.md5(str(paper_data).encode()).hexdigest()
    
    @staticmethod
    def parse_categories(categories: List[str]) -> Dict[str, Any]:
        """Parse and categorize Arxiv categories"""
        try:
            category_info = {
                "primary": categories[0] if categories else None,
                "all_categories": categories,
                "subject_areas": [],
                "is_cs": False,
                "is_math": False,
                "is_physics": False,
                "is_stat": False
            }
            
            # Map categories to subject areas
            subject_mapping = {
                "cs": "Computer Science",
                "math": "Mathematics", 
                "physics": "Physics",
                "stat": "Statistics",
                "q-bio": "Quantitative Biology",
                "q-fin": "Quantitative Finance",
                "econ": "Economics",
                "eess": "Electrical Engineering"
            }
            
            for category in categories:
                subject = category.split(".")[0] if "." in category else category.split("-")[0]
                
                if subject in subject_mapping:
                    subject_area = subject_mapping[subject]
                    if subject_area not in category_info["subject_areas"]:
                        category_info["subject_areas"].append(subject_area)
                
                # Set flags
                if category.startswith("cs."):
                    category_info["is_cs"] = True
                elif category.startswith("math."):
                    category_info["is_math"] = True
                elif category.startswith("physics.") or category.startswith("astro-ph.") or category.startswith("cond-mat."):
                    category_info["is_physics"] = True
                elif category.startswith("stat."):
                    category_info["is_stat"] = True
            
            return category_info
        except Exception as e:
            logger.error(f"Error parsing categories: {str(e)}")
            return {"primary": None, "all_categories": categories, "subject_areas": []}
    
    @staticmethod
    def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
        """Extract potential keywords from text"""
        try:
            if not text:
                return []
            
            # Simple keyword extraction
            # Remove common stop words
            stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                "for", "of", "with", "by", "this", "that", "these", "those",
                "is", "are", "was", "were", "be", "been", "being", "have",
                "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "can", "we", "our", "us", "they",
                "their", "them", "it", "its", "he", "his", "him", "she",
                "her", "hers", "you", "your", "yours", "i", "my", "mine"
            }
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Filter out stop words and count frequency
            word_freq = {}
            for word in words:
                if word not in stop_words and len(word) >= 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    @staticmethod
    def format_paper_for_display(paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format paper data for display"""
        try:
            formatted = {
                "id": paper_data.get("arxiv_id", ""),
                "title": paper_data.get("title", "Untitled"),
                "authors": paper_data.get("authors", []),
                "abstract": paper_data.get("abstract", "")[:500] + "..." if len(paper_data.get("abstract", "")) > 500 else paper_data.get("abstract", ""),
                "categories": paper_data.get("categories", []),
                "published": paper_data.get("published", ""),
                "pdf_url": paper_data.get("pdf_url", ""),
                "entry_id": paper_data.get("entry_id", "")
            }
            
            # Format date
            if formatted["published"]:
                try:
                    pub_date = datetime.fromisoformat(formatted["published"].replace('Z', '+00:00'))
                    formatted["published_formatted"] = pub_date.strftime("%Y-%m-%d")
                except:
                    formatted["published_formatted"] = formatted["published"]
            else:
                formatted["published_formatted"] = "Unknown"
            
            # Format authors
            if len(formatted["authors"]) > 3:
                formatted["authors_display"] = ", ".join(formatted["authors"][:3]) + f" et al. ({len(formatted['authors'])} total)"
            else:
                formatted["authors_display"] = ", ".join(formatted["authors"])
            
            # Extract primary category
            if formatted["categories"]:
                formatted["primary_category"] = formatted["categories"][0]
            else:
                formatted["primary_category"] = "Unknown"
            
            return formatted
        except Exception as e:
            logger.error(f"Error formatting paper: {str(e)}")
            return paper_data
    
    @staticmethod
    def create_paper_summary(paper_data: Dict[str, Any]) -> str:
        """Create a brief summary of a paper"""
        try:
            title = paper_data.get("title", "")
            authors = paper_data.get("authors", [])
            categories = paper_data.get("categories", [])
            
            summary_parts = []
            
            if title:
                summary_parts.append(f"Title: {title}")
            
            if authors:
                if len(authors) <= 3:
                    author_str = ", ".join(authors)
                else:
                    author_str = f"{', '.join(authors[:3])} et al."
                summary_parts.append(f"Authors: {author_str}")
            
            if categories:
                summary_parts.append(f"Categories: {', '.join(categories[:3])}")
            
            return " | ".join(summary_parts)
        except Exception as e:
            logger.error(f"Error creating paper summary: {str(e)}")
            return "Summary unavailable"
    
    @staticmethod
    def batch_validate_papers(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of papers"""
        try:
            validation_results = {
                "total_papers": len(papers),
                "valid_papers": 0,
                "invalid_papers": 0,
                "validation_errors": []
            }
            
            for i, paper in enumerate(papers):
                errors = []
                
                # Check required fields
                if not paper.get("arxiv_id"):
                    errors.append("Missing arxiv_id")
                elif not ArxivUtils.validate_arxiv_id(paper["arxiv_id"]):
                    errors.append("Invalid arxiv_id format")
                
                if not paper.get("title"):
                    errors.append("Missing title")
                
                if not paper.get("abstract"):
                    errors.append("Missing abstract")
                
                if not paper.get("authors"):
                    errors.append("Missing authors")
                
                if errors:
                    validation_results["invalid_papers"] += 1
                    validation_results["validation_errors"].append({
                        "paper_index": i,
                        "arxiv_id": paper.get("arxiv_id", "Unknown"),
                        "errors": errors
                    })
                else:
                    validation_results["valid_papers"] += 1
            
            return validation_results
        except Exception as e:
            logger.error(f"Error validating papers batch: {str(e)}")
            return {"error": str(e)}

# Global instance
arxiv_utils = ArxivUtils()
