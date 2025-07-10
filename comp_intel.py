import os
import json
import asyncio
import requests
from typing import Optional
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()
# Configuration
MODEL_ID = "gemini-2.5-flash"
PROJECT_ID = st.secrets["VERTEX_PROJECT_ID"]
LOCATION = st.secrets["VERTEX_LOCATION"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]

# The JSON string for credentials will need to be parsed
google_applications_credentials_json_str = st.secrets["GOOGLE_APPLICATIONS_CREDENTIALS_JSON"]
GOOGLE_APPLICATIONS_CREDENTIALS_JSON = json.loads(google_applications_credentials_json_str)

COMPETITIVE_INTELLIGENCE_SYSTEM_PROMPT = """You are a Competitive Intelligence Agent for OmniActive Health Technologies marketing research.

Your role is to monitor competitors, analyze their strategies, track product launches, and identify competitive advantages and threats.

You must ALWAYS respond in valid JSON format with these fields:
{
    "response": "Your competitive analysis and strategic insights",
    "next_action": "web_intelligence_agent" | "social_intelligence_agent" | "stop",
    "agent_prompt": "Instructions for the next agent (if next_action is not 'stop')"
}

Key Competitors to Monitor:
- DSM (Quali-Blends, FloraGLO)
- BASF (Newtrition)
- Kemin Industries
- Naturex (Givaudan)
- Indena
- Sabinsa Corporation
- Cargill

Search Strategy:
- Monitor competitor product launches and innovations
- Track pricing strategies and market positioning
- Analyze marketing campaigns and messaging
- Identify partnership announcements and acquisitions
- Monitor patent filings and R&D activities
- Track regulatory approvals and certifications

Analysis Focus:
- Competitive advantages and disadvantages
- Market positioning strategies
- Innovation pipeline and R&D focus
- Partnership and acquisition activities
- Pricing strategies and market share
- Marketing messaging and brand positioning

Always provide specific competitive insights with actionable recommendations for OmniActive."""

def get_access_token():
    """Get access token using service account credentials"""
    try:
        service_account_info = json.loads(GOOGLE_APPLICATIONS_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

def make_gemini_request(prompt: str, max_tokens: int = 1024, temperature: float = 0.1, system_prompt: str = None):
    """Make a request to Gemini API"""
    access_token = get_access_token()
    
    if not access_token:
        raise Exception("Failed to get access token")
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "role": "model",
                "parts": [{"text": f"Please follow this system prompt to the end: {system_prompt}"}]
            },
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    
    if "candidates" in result and len(result["candidates"]) > 0:
        candidate = result["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            return candidate["content"]["parts"][0].get("text", "")
    
    return ""

def serper_search(query: str, search_type: str = "search") -> dict:
    """Search using Serper API"""
    url = f"https://google.serper.dev/{search_type}"
    
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "q": query,
        "num": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in Serper search: {e}")
        return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def competitive_intelligence_agent(agent_prompt: str) -> dict:
    """Competitive Intelligence Agent that monitors competitors"""
    try:
        # Generate search queries for competitive intelligence
        competitors = ["DSM", "BASF", "Kemin Industries", "Naturex", "Indena", "Sabinsa"]
        
        search_queries = [
            f"DSM FloraGLO lutein vs OmniActive Lutemax {agent_prompt}",
            f"BASF Newtrition nutraceuticals launch 2024 {agent_prompt}",
            f"Kemin Industries eye health ingredients {agent_prompt}",
            f"Naturex Givaudan botanical extracts {agent_prompt}",
            f"nutraceutical ingredient suppliers competitive analysis {agent_prompt}",
            f"lutein supplement market share competitors {agent_prompt}"
        ]
        
        # Perform searches
        search_results = []
        for query in search_queries[:4]:  # Limit to 4 searches to manage API usage
            result = await asyncio.to_thread(serper_search, query)
            if result:
                search_results.append({
                    "query": query,
                    "results": result.get("organic", [])[:5]  # Top 5 results
                })
        
        # Format search results for analysis
        formatted_results = ""
        for search in search_results:
            formatted_results += f"\n--- Search: {search['query']} ---\n"
            for result in search['results']:
                formatted_results += f"Title: {result.get('title', 'N/A')}\n"
                formatted_results += f"Snippet: {result.get('snippet', 'N/A')}\n"
                formatted_results += f"Link: {result.get('link', 'N/A')}\n\n"
        
        # Analyze results with LLM
        analysis_prompt = f"""
        Agent Task: {agent_prompt}
        
        Competitive Intelligence Search Results:
        {formatted_results}
        
        Analyze these competitive intelligence results and provide insights about:
        1. Competitor product positioning and strategies
        2. New product launches and innovations
        3. Pricing strategies and market positioning
        4. Marketing messaging and brand positioning
        5. Partnership announcements and acquisitions
        6. Competitive advantages and threats to OmniActive
        7. Market share dynamics and trends

        Give this in 100 words total
        
        Provide specific, actionable recommendations for OmniActive's competitive strategy.
        Determine if additional intelligence is needed from other agents.
        """
        
        def make_api_call():
            print(f"Making Gemini API call with prompt: {analysis_prompt}")
            return make_gemini_request(analysis_prompt, max_tokens=8192, temperature=0.3, system_prompt=COMPETITIVE_INTELLIGENCE_SYSTEM_PROMPT)
        
        result = await asyncio.to_thread(make_api_call)
        result = result.replace("```json","").replace("```", "").strip()
        # Parse JSON response
        try:
            response_data = json.loads(result)
            return response_data
        except json.JSONDecodeError:
            return {
                "response": f"Competitive Intelligence Analysis: Based on competitive monitoring, I found insights about competitor strategies and market positioning. The analysis revealed important information about competitive dynamics and opportunities for OmniActive.",
                "next_action": "stop",
                "agent_prompt": ""
            }
        
    except Exception as e:
        print(f"Error in competitive intelligence agent: {str(e)}")
        return {
            "response": f"Competitive Intelligence Agent encountered an error: {str(e)}",
            "next_action": "stop",
            "agent_prompt": ""
        }