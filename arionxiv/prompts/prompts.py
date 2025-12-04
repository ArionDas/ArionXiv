"""
Centralized Prompt Templates for ArionXiv

Prompts are stored in MongoDB and fetched on-demand with TTL caching.
Admin can edit prompts directly in the database (prompts collection).

Fallback prompts are used when database is unavailable.
"""

from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# FALLBACK PROMPTS - Used when database is unavailable
# Master prompts are stored in MongoDB (prompts collection)
# ============================================================================

DEFAULT_PROMPTS: Dict[str, str] = {
    
    "comprehensive_paper_analysis": """You are an expert research analyst. Analyze this research paper thoroughly.

Paper Content:
{content}

IMPORTANT: You MUST respond with ONLY valid JSON. No explanations, no markdown, no text before or after the JSON.

Return EXACTLY this JSON structure (replace placeholders with your analysis):
{{
  "summary": "A comprehensive 3-4 sentence summary of the paper's main contribution and findings",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4"],
  "methodology": "Description of the research methodology, algorithms, or experimental approach used",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "limitations": ["Limitation 1", "Limitation 2"],
  "technical_details": "Detailed technical explanation of the core innovations, architectures, or implementations",
  "broader_impact": "Analysis of the paper's broader implications, applications, and future research directions",
  "confidence_score": 0.85,
  "relevance_tags": ["tag1", "tag2", "tag3"],
  "technical_level": "intermediate",
  "novelty_score": 0.8,
  "impact_potential": "Description of the potential real-world impact"
}}

Remember: Output ONLY the JSON object. No other text.""",

    "enhanced_paper_analysis": """You are an elite research analyst. Provide an exceptionally thorough analysis of this research paper.

Paper Content:
{content}

Provide a detailed analysis in JSON format with: summary, key_findings, methodology, strengths, limitations, technical_details, broader_impact, confidence_score, relevance_tags, technical_level.""",

    "paper_analysis": """Analyze this research paper comprehensively.

Title: {title}
Abstract: {abstract}
Content: {content}

Provide JSON with: summary, key_findings, methodology, strengths, limitations.""",

    "quick_analysis": """Provide a quick analysis of this paper.

Title: {title}
Abstract: {abstract}

Return JSON with: summary, main_contribution, relevance_score.""",

    "research_analysis": """Analyze this research paper in depth.

Title: {title}
Content: {content}

Provide JSON with: detailed_summary, methodology, findings, implications.""",

    "summary_analysis": """Provide a summary analysis of this paper.

Content: {content}

Focus on key points and main contributions.""",

    "detailed_analysis": """Provide a detailed analysis of this paper.

Content: {content}

Include methodology, findings, strengths, and limitations.""",

    "technical_analysis": """Provide a technical analysis of this paper.

Content: {content}

Focus on technical contributions, algorithms, and implementation details.""",

    "insights_analysis": """Extract key insights from this paper.

Content: {content}

Provide actionable insights and implications for the field.""",

    "daily_dose_analysis": """Analyze this research paper and provide a thorough analysis.

Title: {title}

Authors: {authors}

Categories: {categories}

Abstract: {abstract}

Provide a structured analysis with the following sections:
1. SUMMARY: A concise 2-3 sentence summary of the paper's main contribution.
2. KEY FINDINGS: List the 3-4 most important findings or contributions.
3. METHODOLOGY: Brief description of the approach or methods used.
4. SIGNIFICANCE: Why this paper matters and its potential impact.
5. LIMITATIONS: Any noted limitations or areas for future work.
6. RELEVANCE SCORE: Rate from 1-10 how impactful this paper is for practitioners.

Be concise but thorough. Focus on actionable insights.""",

    "trend_analysis": """You are a research trend analyst. Based on the following research papers, generate insights about current trends and directions.

Papers:
{papers_data}

Analyze: emerging trends, common methodologies, novel approaches, cross-paper themes, future opportunities, field evolution.""",

    "enhanced_trend_analysis": """You are an expert research trend analyst. Analyze the following papers for emerging trends and patterns.

Papers:
{papers_data}

Provide JSON with: emerging_trends, common_themes, methodology_patterns, future_directions, key_insights.""",

    "paper_summary": """Summarize the following collection of research papers concisely.

Papers:
{papers_data}

Provide a brief, coherent summary of the main themes and contributions.""",

    "personalized_recommendations": """Based on the user's research interests and history, recommend relevant research directions.

User Profile:
{user_profile}

Recent Activity:
{recent_activity}

Provide JSON with: recommended_topics, suggested_papers, emerging_areas, learning_path.""",

    "rag_chat": """You are an AI research assistant helping users understand research papers. Your role is to provide accurate, helpful answers based on the paper content.

RELEVANT SECTIONS FROM THE PAPER:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {message}

Instructions:
- Provide comprehensive, detailed answers using specific details from the paper
- Quote relevant passages when appropriate
- If the answer is not in the provided context, say so clearly
- Be conversational but maintain technical accuracy
- Structure longer answers with clear sections""",

}


# ============================================================================
# PROMPT SERVICE INTEGRATION
# ============================================================================

try:
    from ..services.unified_prompt_service import unified_prompt_service
    PROMPT_SERVICE_AVAILABLE = True
except ImportError:
    PROMPT_SERVICE_AVAILABLE = False
    logger.warning("UnifiedPromptService not available - using fallback prompts")

def format_prompt(prompt_name: str, **kwargs) -> str:
    """
    Format a prompt template with provided variables.
    Fetches from MongoDB with caching, falls back to local prompts if unavailable.
    """
    if PROMPT_SERVICE_AVAILABLE:
        try:
            try:
                loop = asyncio.get_running_loop()
                raise RuntimeError("In async context")
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if not loop.is_running():
                    formatted = loop.run_until_complete(
                        unified_prompt_service.format_prompt(prompt_name, **kwargs)
                    )
                    return formatted
            
        except Exception as e:
            logger.debug(f"Using fallback prompt for {prompt_name}: {str(e)}")
    
    # Fallback to local prompts
    if prompt_name in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[prompt_name].format(**kwargs)
    else:
        logger.warning(f"Unknown prompt '{prompt_name}', using generic template")
        return f"Process the following with context: {kwargs}"

async def format_prompt_async(prompt_name: str, **kwargs) -> str:
    """
    Async version of format_prompt for use in async contexts.
    """
    if PROMPT_SERVICE_AVAILABLE:
        try:
            return await unified_prompt_service.format_prompt(prompt_name, **kwargs)
        except Exception as e:
            logger.debug(f"Using fallback prompt for {prompt_name}: {str(e)}")
    
    # Fallback to local prompts
    if prompt_name in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[prompt_name].format(**kwargs)
    else:
        logger.warning(f"Unknown prompt '{prompt_name}', using generic template")
        return f"Process the following with context: {kwargs}"


def get_all_prompts() -> Dict[str, str]:
    """Get all available prompt names"""
    return DEFAULT_PROMPTS
