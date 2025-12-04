"""
FastAPI server for ArionXiv package
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
import logging

from .services.unified_database_service import unified_database_service
from .services.unified_auth_service import verify_token
from .services.unified_paper_service import unified_paper_service
from .services.unified_user_service import unified_user_service
from .services.unified_analysis_service import unified_analysis_service
from .arxiv_operations.client import ArxivClient
from .arxiv_operations.searcher import ArxivSearcher
from .arxiv_operations.fetcher import ArxivFetcher

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ArionXiv API",
    description="AI-powered research paper ingestion pipeline with user authentication and daily analysis",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
arxiv_client = ArxivClient()
arxiv_searcher = ArxivSearcher()
arxiv_fetcher = ArxivFetcher()

# Startup event to initialize database connections
@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    try:
        logger.info("Starting ArionXiv API server")
        await asyncio.wait_for(unified_database_service.connect_mongodb(), timeout=10.0)
        logger.info("ArionXiv API server started successfully")
    except asyncio.TimeoutError:
        logger.error("Database connection timeout during startup")
        raise Exception("Database connection timeout")
    except Exception as e:
        logger.error(f"Failed to start ArionXiv API server: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    try:
        await unified_database_service.disconnect()
        logger.info("ArionXiv API server shut down gracefully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to ArionXiv API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/user/profile")
async def get_user_profile(current_user: Dict = Depends(verify_token)):
    """Get current user profile"""
    try:
        logger.debug(f"Fetching profile for user: {current_user['email']}")
        user = await unified_user_service.get_user_by_email(current_user["email"])
        if not user:
            logger.warning(f"User not found: {current_user['email']}")
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.debug(f"Profile fetched for user: {current_user['email']}")
        return {"user": user}
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# Paper Management Endpoints
@app.get("/papers/search")
async def search_papers(
    query: str,
    max_results: int = 10,
    category: Optional[str] = None,
    current_user: Dict = Depends(verify_token)
):
    """Search for papers on arXiv"""
    try:
        logger.info(f"Searching papers: query='{query}', max_results={max_results}, category={category}")
        papers = await arxiv_searcher.search_papers(
            query=query,
            max_results=max_results,
            category=category
        )
        logger.info(f"Paper search completed: {len(papers)} results found")
        return {
            "papers": papers,
            "count": len(papers),
            "query": query
        }
    except Exception as e:
        logger.error(f"Paper search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/papers/{arxiv_id}/fetch")
async def fetch_paper(
    arxiv_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Fetch and store a specific paper"""
    try:
        logger.info(f"Fetching paper: {arxiv_id}")
        paper = await arxiv_fetcher.fetch_paper(arxiv_id)
        
        if not paper:
            logger.warning(f"Paper not found: {arxiv_id}")
            raise HTTPException(status_code=404, detail="Paper not found")
        
        stored_paper = await unified_paper_service.store_paper(paper)
        logger.info(f"Paper stored successfully: {arxiv_id}")
        
        return {
            "message": "Paper fetched successfully",
            "paper": stored_paper
        }
    except Exception as e:
        logger.error(f"Paper fetch failed for {arxiv_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch paper")

@app.post("/papers/{paper_id}/analyze")
async def analyze_paper(
    paper_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Analyze a stored paper"""
    try:
        logger.info(f"Starting paper analysis: {paper_id}")
        paper = await unified_paper_service.get_paper_by_id(paper_id)
        
        if not paper:
            logger.warning(f"Paper not found for analysis: {paper_id}")
            raise HTTPException(status_code=404, detail="Paper not found")
        
        analysis = await unified_analysis_service.analyze_paper(paper)
        logger.info(f"Paper analysis completed: {paper_id}")
        
        return {
            "message": "Paper analyzed successfully",
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Paper analysis failed for {paper_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/papers/user")
async def get_user_papers(
    current_user: Dict = Depends(verify_token),
    limit: int = 20,
    skip: int = 0
):
    """Get papers associated with current user"""
    try:
        logger.debug(f"Fetching papers for user: {current_user['email']} (limit={limit}, skip={skip})")
        papers = await unified_paper_service.get_user_papers(
            user_email=current_user["email"],
            limit=limit,
            skip=skip
        )
        logger.debug(f"Retrieved {len(papers)} papers for user: {current_user['email']}")
        
        return {
            "papers": papers,
            "count": len(papers)
        }
    except Exception as e:
        logger.error(f"Failed to get user papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve papers")

# Daily Analysis Endpoints
@app.get("/analysis/daily")
async def get_daily_analysis(
    date: Optional[str] = None,
    current_user: Dict = Depends(verify_token)
):
    """Get daily analysis for a specific date"""
    try:
        if date:
            analysis_date = datetime.fromisoformat(date)
        else:
            analysis_date = datetime.utcnow()
        
        logger.info(f"Fetching daily analysis for date: {analysis_date.isoformat()}")
        analysis = await unified_analysis_service.get_daily_analysis(analysis_date)
        logger.debug(f"Daily analysis retrieved for: {analysis_date.isoformat()}")
        
        return {
            "analysis": analysis,
            "date": analysis_date.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get daily analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@app.post("/analysis/daily/trigger")
async def trigger_daily_analysis(current_user: Dict = Depends(verify_token)):
    """Manually trigger daily analysis"""
    try:
        logger.info("Manually triggering daily analysis")
        analysis_task = asyncio.create_task(
            unified_analysis_service.run_daily_analysis()
        )
        logger.info("Daily analysis task started")
        
        return {
            "message": "Daily analysis triggered",
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Failed to trigger daily analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to trigger analysis")


async def main():
    """Main function to run the server"""
    import uvicorn
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    logger.info("Starting ArionXiv API server on http://0.0.0.0:8000")
    logger.info("API Documentation: http://0.0.0.0:8000/docs")
    logger.info("Health Check: http://0.0.0.0:8000/health")
    
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())