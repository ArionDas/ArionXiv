#!/usr/bin/env python3
"""
Deployment verification script for ArionXiv
Checks all critical imports and configurations
"""

import sys
import os

def verify_imports():
    """Verify all critical imports"""
    print("🔍 Verifying imports...")
    
    try:
        from arionxiv.server import app
        print("✓ Server module")
        
        from arionxiv.services.unified_auth_service import unified_auth_service
        print("✓ Auth service")
        
        from arionxiv.services.unified_database_service import unified_database_service
        print("✓ Database service")
        
        from arionxiv.services.unified_paper_service import unified_paper_service
        print("✓ Paper service")
        
        from arionxiv.services.unified_analysis_service import unified_analysis_service, rag_chat_system
        print("✓ Analysis service")
        print("✓ RAG chat system")
        
        from arionxiv.services.llm_inference import groq_client, openrouter_client
        print("✓ LLM clients")
        
        from arionxiv.arxiv_operations import client, searcher, fetcher
        print("✓ ArXiv operations")
        
        from arionxiv.utils.api_helpers import sanitize_arxiv_id
        print("✓ API helpers")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def check_env_vars():
    """Check critical environment variables"""
    print("\n🔍 Checking environment variables...")
    
    required = {
        "MONGODB_URI": "Database connection",
        "JWT_SECRET_KEY": "Authentication",
    }
    
    recommended = {
        "OPENROUTER_API_KEY": "OpenRouter LLM (or GROQ_API_KEY)",
        "GROQ_API_KEY": "Groq LLM (or OPENROUTER_API_KEY)",
        "GEMINI_API_KEY": "Google Gemini for embeddings",
    }
    
    missing_required = []
    missing_recommended = []
    
    for key, purpose in required.items():
        if os.getenv(key):
            print(f"✓ {key} ({purpose})")
        else:
            print(f"✗ {key} ({purpose}) - REQUIRED")
            missing_required.append(key)
    
    for key, purpose in recommended.items():
        if os.getenv(key):
            print(f"✓ {key} ({purpose})")
        else:
            print(f"○ {key} ({purpose}) - optional but recommended")
            missing_recommended.append(key)
    
    # Check that at least one LLM provider is available
    has_llm = os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY")
    if not has_llm:
        print("\n⚠ WARNING: No LLM API key found (need OPENROUTER_API_KEY or GROQ_API_KEY)")
        missing_required.append("LLM_API_KEY")
    
    return len(missing_required) == 0

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("ArionXiv Deployment Verification")
    print("=" * 60)
    
    imports_ok = verify_imports()
    env_ok = check_env_vars()
    
    print("\n" + "=" * 60)
    if imports_ok and env_ok:
        print("✓ All checks passed - Ready for deployment!")
        return 0
    else:
        print("✗ Some checks failed - Review errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
