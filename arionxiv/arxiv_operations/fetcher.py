import requests
import aiohttp
import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ..services.unified_pdf_service import pdf_processor

logger = logging.getLogger(__name__)

class ArxivFetcher:
    """Fetches and processes papers from Arxiv"""
    
    def __init__(self):
        self.session = None
        self.download_dir = "downloads"
        os.makedirs(self.download_dir, exist_ok=True)
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def fetch_paper_pdf(self, arxiv_id: str, pdf_url: str) -> Optional[str]:
        """Fetch PDF for a paper"""
        try:
            session = await self._get_session()
            
            # Clean arxiv_id for filename
            safe_id = arxiv_id.replace('/', '_').replace(':', '_')
            pdf_path = os.path.join(self.download_dir, f"{safe_id}.pdf")
            
            # Check if already downloaded
            if os.path.exists(pdf_path):
                logger.info(f"PDF already exists: {pdf_path}")
                return pdf_path
            
            # Download PDF
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(pdf_path, 'wb') as f:
                        f.write(content)
                    logger.info(f"Downloaded PDF: {pdf_path}")
                    return pdf_path
                else:
                    logger.error(f"Failed to download PDF: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching PDF for {arxiv_id}: {str(e)}")
            return None
    
    async def fetch_and_process_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch and process a complete paper"""
        try:
            arxiv_id = paper_data.get("arxiv_id")
            pdf_url = paper_data.get("pdf_url")
            
            logger.info(f"Processing paper: {arxiv_id}")
            
            if not arxiv_id or not pdf_url:
                logger.warning(f"Missing arxiv_id or pdf_url for paper")
                return {"error": "Missing arxiv_id or pdf_url"}
            
            # Fetch PDF
            logger.debug(f"Fetching PDF from: {pdf_url}")
            pdf_path = await self.fetch_paper_pdf(arxiv_id, pdf_url)
            if not pdf_path:
                logger.error(f"Failed to download PDF for {arxiv_id}")
                return {"error": "Failed to download PDF"}
            
            # Process PDF
            logger.debug(f"Processing PDF: {pdf_path}")
            processing_result = await pdf_processor.process_pdf(pdf_path)
            
            logger.info(f"Successfully processed paper: {arxiv_id}")
            
            # Combine paper metadata with processed content
            result = {
                **paper_data,
                "pdf_path": pdf_path,
                "processed_content": processing_result,
                "fetch_timestamp": asyncio.get_event_loop().time()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            return {"error": str(e)}
    
    async def batch_fetch_papers(self, papers: list) -> list:
        """Fetch multiple papers concurrently"""
        try:
            logger.info(f"Starting batch fetch for {len(papers)} papers")
            tasks = []
            for paper in papers:
                task = self.fetch_and_process_paper(paper)
                tasks.append(task)
            
            # Limit concurrent downloads
            semaphore = asyncio.Semaphore(3)
            
            async def bounded_fetch(paper):
                async with semaphore:
                    return await self.fetch_and_process_paper(paper)
            
            bounded_tasks = [bounded_fetch(paper) for paper in papers]
            results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
            
            # Filter out exceptions
            successful_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch fetch error: {str(result)}", exc_info=True)
                else:
                    successful_results.append(result)
            
            logger.info(f"Batch fetch completed: {len(successful_results)}/{len(papers)} successful")
            return successful_results
        except Exception as e:
            logger.error(f"Batch fetch error: {str(e)}", exc_info=True)
            return []
    
    def fetch_paper_sync(self, arxiv_id: str, pdf_url: str) -> Optional[str]:
        """Synchronous version of PDF fetch"""
        try:
            safe_id = arxiv_id.replace('/', '_').replace(':', '_')
            pdf_path = os.path.join(self.download_dir, f"{safe_id}.pdf")
            
            if os.path.exists(pdf_path):
                return pdf_path
            
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded PDF: {pdf_path}")
                return pdf_path
            else:
                logger.error(f"Failed to download PDF: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching PDF sync for {arxiv_id}: {str(e)}")
            return None
    
    async def cleanup_downloads(self, max_age_days: int = 7):
        """Clean up old downloaded files"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            for filename in os.listdir(self.download_dir):
                file_path = os.path.join(self.download_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None

# Global instance
arxiv_fetcher = ArxivFetcher()
