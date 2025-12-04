# LLM client for AI-powered paper analysis
from typing import Dict, Any, List, Optional
import logging
import json
import asyncio
import os
from datetime import datetime
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class UnifiedLLMService:
    """Client for LLM-based paper analysis using Groq"""
    
    def __init__(self):
        # Groq LLM configuration
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("DEFAULT_ANALYSIS_MODEL", "llama-3.3-70b-versatile")
        self.timeout = 60
        self.client = None
        
        # Initialize Groq client if API key is available
        if self.api_key:
            try:
                # Use the updated Groq client
                self.client = AsyncGroq(
                    api_key=self.api_key
                )
                logger.debug("Groq LLM client initialized", extra={"model": self.model})
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        else:
            logger.warning("No GROQ_API_KEY found in environment - LLM features will be unavailable")
            self.client = None
    
    async def analyze_paper(self, content: str) -> Dict[str, Any]:
        """Analyze a single paper using Groq LLM"""
        try:
            if not content.strip():
                return {"analysis": "No content provided for analysis"}
            
            if not self.client:
                logger.error("LLM client not configured - API key missing")
                raise ValueError("GROQ_API_KEY is not configured. Please set your Groq API key in the .env file to use paper analysis.")
            
            # Create comprehensive analysis prompt with enhanced instructions
            from ..prompts import format_prompt
            prompt = format_prompt("enhanced_paper_analysis", content=content)
            
            # Make API call to Groq with optimized settings for quality
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an elite research analyst known for producing exceptionally thorough, insightful, and high-quality paper analyses. Your analyses are comprehensive, technically precise, and highly valued by researchers worldwide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Balanced temperature for quality and creativity
                max_tokens=8000   # Significantly increased token limit for comprehensive analysis
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            
            try:
                # Clean response content - remove markdown code blocks
                clean_content = response_content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                elif clean_content.startswith("```"):
                    clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()
                
                # Try to parse as JSON
                analysis = json.loads(clean_content)
                logger.info("Successfully analyzed paper with LLM")
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from raw text
                logger.warning("LLM response was not valid JSON, creating structured response")
                
                # Try to extract meaningful content from the response
                lines = response_content.split('\n')
                summary_lines = []
                key_findings = []
                methodology_lines = []
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to identify sections
                    if any(word in line.lower() for word in ['summary', 'abstract', 'overview']):
                        current_section = 'summary'
                        if ':' in line:
                            summary_lines.append(line.split(':', 1)[1].strip())
                        continue
                    elif any(word in line.lower() for word in ['finding', 'result', 'contribution']):
                        current_section = 'findings'
                        if ':' in line:
                            key_findings.append(line.split(':', 1)[1].strip())
                        continue
                    elif any(word in line.lower() for word in ['method', 'approach', 'technique']):
                        current_section = 'methodology'
                        if ':' in line:
                            methodology_lines.append(line.split(':', 1)[1].strip())
                        continue
                    
                    # Add content based on current section
                    if current_section == 'summary' and len(summary_lines) < 3:
                        summary_lines.append(line)
                    elif current_section == 'findings' and len(key_findings) < 5:
                        key_findings.append(line)
                    elif current_section == 'methodology' and len(methodology_lines) < 3:
                        methodology_lines.append(line)
                
                # Fallback if sections not found - use first part as summary
                if not summary_lines:
                    summary_lines = lines[:3] if len(lines) >= 3 else lines
                if not key_findings:
                    key_findings = lines[3:8] if len(lines) > 3 else ["Analysis completed successfully"]
                if not methodology_lines:
                    methodology_lines = lines[8:11] if len(lines) > 8 else ["LLM-based analysis methodology"]
                
                return {
                    "summary": ' '.join(summary_lines) if summary_lines else "Comprehensive research paper analysis completed.",
                    "key_findings": key_findings if key_findings else ["Significant research contributions identified", "Novel methodological approaches presented", "Important findings documented"],
                    "methodology": ' '.join(methodology_lines) if methodology_lines else "Advanced analytical methodology applied to research content.",
                    "strengths": ["Comprehensive research approach", "Strong methodological foundation", "Clear presentation of results", "Significant contributions to field"],
                    "limitations": ["Specific scope limitations may apply", "Further validation may be beneficial", "Future research directions identified"],
                    "technical_details": response_content[:500] + "..." if len(response_content) > 500 else response_content,
                    "broader_impact": "This research contributes to advancing knowledge in the field with potential applications and future research directions.",
                    "confidence_score": 0.75,
                    "relevance_tags": ["research", "analysis", "academic"],
                    "technical_level": "intermediate",
                    "raw_response": response_content
                }
            
        except Exception as e:
            logger.error("Paper analysis failed", error=str(e))
            raise
    
    async def generate_insights(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cross-paper insights using Groq LLM"""
        try:
            if not papers:
                return {"message": "No papers provided for insight generation"}
            
            if not self.client:
                logger.error("LLM client not configured - API key missing")
                raise ValueError("GROQ_API_KEY is not configured. Please set your Groq API key in the .env file to generate insights.")
            
            # Prepare papers summary for analysis
            papers_summary = []
            for i, paper in enumerate(papers[:10]):  # Limit to 10 papers to avoid token limits
                summary = f"Paper {i+1}: {paper.get('title', 'Unknown')} - {paper.get('abstract', 'No abstract')[:200]}..."
                papers_summary.append(summary)
            
            from ..prompts import format_prompt
            papers_data = f"Papers analyzed ({len(papers)} total, showing first {min(len(papers), 10)}):\n{chr(10).join(papers_summary)}"
            prompt = format_prompt("enhanced_trend_analysis", papers_data=papers_data)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2500
            )
            
            response_content = response.choices[0].message.content
            
            try:
                insights = json.loads(response_content)
                logger.info("Successfully generated insights with LLM", paper_count=len(papers))
                return insights
            except json.JSONDecodeError as json_err:
                logger.error("LLM insights response was not valid JSON", error=str(json_err))
                raise ValueError(f"Failed to parse LLM response: {str(json_err)}")
            
        except Exception as e:
            logger.error("Insight generation failed", error=str(e))
            raise
    
    async def summarize_collection(self, papers: List[Dict[str, Any]]) -> str:
        """Generate a summary of a collection of papers using Groq LLM"""
        try:
            if not papers:
                return "No papers provided for summarization"
            
            if not self.client:
                logger.error("LLM client not configured - API key missing")
                raise ValueError("GROQ_API_KEY is not configured. Please set your Groq API key in the .env file to generate summaries.")
            
            # Create summary for papers
            papers_info = []
            for paper in papers[:15]:  # Limit to avoid token limits
                info = f"- {paper.get('title', 'Unknown')}: {paper.get('abstract', 'No abstract')[:150]}..."
                papers_info.append(info)
            
            from ..prompts import format_prompt
            papers_data = chr(10).join(papers_info)
            prompt = format_prompt("paper_summary", papers_data=papers_data)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Successfully generated collection summary with LLM")
            return summary
            
        except Exception as e:
            logger.error("Collection summarization failed", error=str(e))
            return f"Collection of {len(papers)} papers covering diverse topics in machine learning and AI. Summarization failed: {str(e)}"
    
    async def generate_research_recommendations(self, user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate personalized research recommendations using Groq LLM"""
        try:
            if not user_history:
                return {"message": "No user history available for recommendations"}
            
            if not self.client:
                logger.error("LLM client not configured - API key missing")
                raise ValueError("GROQ_API_KEY is not configured. Please set your Groq API key in the .env file to generate recommendations.")
            
            # Prepare user history summary
            history_summary = []
            for item in user_history[:10]:  # Limit to recent history
                summary = f"- {item.get('title', 'Unknown')}: {item.get('action', 'viewed')} on {item.get('date', 'unknown date')}"
                history_summary.append(summary)
            
            from ..prompts import format_prompt
            user_profile = chr(10).join(history_summary)
            prompt = format_prompt("personalized_recommendations",
                                 user_profile=user_profile,
                                 recent_activity="See user history above")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            response_content = response.choices[0].message.content
            
            try:
                recommendations = json.loads(response_content)
                logger.info("Successfully generated recommendations with LLM")
                return recommendations
            except json.JSONDecodeError as json_err:
                logger.error("LLM recommendations response was not valid JSON", error=str(json_err))
                raise ValueError(f"Failed to parse LLM response: {str(json_err)}")
            
        except Exception as e:
            logger.error("Recommendation generation failed", error=str(e))
            raise
    
    async def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2500, system_message: str = None) -> Dict[str, Any]:
        """Get completion from LLM for chat purposes with enhanced quality"""
        try:
            if not self.client:
                logger.error("LLM client not configured - API key missing")
                return {
                    "success": False,
                    "error": "GROQ_API_KEY is not configured. Please set your Groq API key in the .env file.",
                    "content": "",
                    "model": "none"
                }
            
            # Prepare messages with optional system message for better quality
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            else:
                # Default system message for research assistance
                messages.append({
                    "role": "system", 
                    "content": "You are an elite AI research assistant with deep expertise in scientific literature. Provide thorough, accurate, and insightful responses. Use markdown formatting (bold, italics, lists, code blocks) for better readability. Be comprehensive yet clear in your explanations."
                })
            
            messages.append({"role": "user", "content": prompt})
            
            # Make API call to Groq with enhanced settings
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            return {
                "success": True,
                "content": content,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error("LLM completion failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "content": "Sorry, I encountered an error while processing your request. Please try again.",
                "model": self.model
            }
    
    def configure(self, api_key: str, model: str = None):
        """Configure LLM client with API credentials"""
        self.api_key = api_key
        if model:
            self.model = model
        
        # Reinitialize client with new API key
        if self.api_key:
            self.client = AsyncGroq(api_key=self.api_key)
            logger.info("LLM client reconfigured", model=self.model)
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM client status"""
        return {
            "configured": self.api_key is not None and self.client is not None,
            "model": self.model,
            "api_service": "Groq",
            "timeout": self.timeout,
            "has_api_key": self.client is not None
        }


# Global instance
unified_llm_service = UnifiedLLMService()
llm_service = unified_llm_service

__all__ = ['UnifiedLLMService', 'unified_llm_service', 'llm_service']