"""
File Cleanup Utility for ArionXiv
Manages temporary paper downloads and cleanup after usage
"""

import os
from pathlib import Path
from typing import List, Optional
import logging

from ..arxiv_operations.utils import ArxivUtils

logger = logging.getLogger(__name__)

class FileCleanupManager:
    """Manages cleanup of downloaded paper files"""
    
    def __init__(self):
        # Get the downloads directory
        self.downloads_dir = self._get_downloads_dir()
    
    def _get_downloads_dir(self) -> Path:
        """Get the downloads directory path"""
        # Default to project root downloads directory
        project_root = Path(__file__).parent.parent.parent
        downloads_dir = project_root / "downloads"
        downloads_dir.mkdir(exist_ok=True)
        return downloads_dir
    
    def cleanup_paper_files(self, paper_id: str) -> bool:
        """
        Delete all files associated with a paper ID
        
        Args:
            paper_id: ArXiv paper ID (e.g., "1706.03762")
            
        Returns:
            bool: True if cleanup was successful
        """
        try:
            # Clean version IDs from paper_id using normalized function
            clean_id = ArxivUtils.normalize_arxiv_id(paper_id)
            
            deleted_count = 0
            
            # Find all files matching the paper ID pattern
            patterns = [
                f"{clean_id}*.pdf",
                f"{clean_id}*.txt",
                f"*{clean_id}*.pdf",
                f"*{clean_id}*.txt"
            ]
            
            for pattern in patterns:
                matching_files = list(self.downloads_dir.glob(pattern))
                for file_path in matching_files:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} files for paper {paper_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup files for paper {paper_id}: {e}")
            return False
    
    def cleanup_multiple_papers(self, paper_ids: List[str]) -> int:
        """
        Cleanup files for multiple papers
        
        Args:
            paper_ids: List of paper IDs to cleanup
            
        Returns:
            int: Number of papers successfully cleaned up
        """
        success_count = 0
        for paper_id in paper_ids:
            if self.cleanup_paper_files(paper_id):
                success_count += 1
        
        return success_count
    
    def get_paper_files(self, paper_id: str) -> List[Path]:
        """
        Get all files associated with a paper ID
        
        Args:
            paper_id: ArXiv paper ID
            
        Returns:
            List[Path]: List of file paths for the paper
        """
        try:
            # Clean version IDs from paper_id using normalized function
            clean_id = ArxivUtils.normalize_arxiv_id(paper_id)
            
            files = []
            
            # Find all files matching the paper ID pattern
            patterns = [
                f"{clean_id}*.pdf",
                f"{clean_id}*.txt",
                f"*{clean_id}*.pdf",
                f"*{clean_id}*.txt"
            ]
            
            for pattern in patterns:
                matching_files = list(self.downloads_dir.glob(pattern))
                files.extend(matching_files)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to get files for paper {paper_id}: {e}")
            return []
    
    def cleanup_all_downloads(self) -> int:
        """
        Clean up all downloaded files
        
        Returns:
            int: Number of files deleted
        """
        try:
            deleted_count = 0
            
            for file_path in self.downloads_dir.glob("*"):
                if file_path.is_file() and file_path.suffix in ['.pdf', '.txt']:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path.name}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} total files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup all downloads: {e}")
            return 0

# Global instance
file_cleanup_manager = FileCleanupManager()
