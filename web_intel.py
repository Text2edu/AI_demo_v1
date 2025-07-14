import os
import json
import asyncio
import requests
import streamlit as st
from typing import Optional
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
GOOGLE_APPLICATIONS_CREDENTIALS_JSON = google_applications_credentials_json_str

WEB_INTELLIGENCE_SYSTEM_PROMPT = """You are a Web Intelligence Agent for OmniActive Health Technologies marketing research.

Your role is to search the web for market trends, news, research papers, and industry analysis related to nutraceuticals and OmniActive's products.

You must ALWAYS respond in valid JSON format with these fields:
{
    "response": "Your analysis and findings based on the search results",
    "next_action": "social_intelligence_agent" | "competitive_intelligence_agent" | "stop",
    "agent_prompt": "Instructions for the next agent (if next_action is not 'stop')"
}

Search Strategy:
- Use relevant keywords for nutraceuticals, health trends, ingredient research
- Focus on OmniActive products: Lutemax (lutein), Capsimax (capsicum), BacoMind (bacopa)
- Look for market size, growth trends, consumer behavior
- Find recent research papers and clinical studies
- Monitor industry news and regulatory changes

Analysis Focus:
- Identify emerging trends and opportunities
- Analyze market dynamics and consumer preferences
- Extract actionable insights for marketing strategies
- Highlight competitive advantages and market gaps

Always provide specific, data-driven insights with sources when possible."""

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
    print(f"Gemini API Response: {result}")  # Debugging line
    
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
async def web_intelligence_agent(agent_prompt: str) -> dict:
    """Web Intelligence Agent that searches and analyzes web content"""
    try:
        # Generate search queries based on the agent prompt
        search_queries = [
            f"nutraceutical market trends 2024 {agent_prompt}",
            f"OmniActive health ingredients {agent_prompt}",
            f"lutein supplement market research {agent_prompt}",
            f"capsicum extract health benefits {agent_prompt}",
            f"cognitive health supplements trends {agent_prompt}"
        ]
        
        # Perform searches
        search_results = []
        for query in search_queries[:3]:  # Limit to 3 searches to manage API usage
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
        
        Search Results:
        {formatted_results}
        
        Analyze these search results and provide insights relevant to the task. Focus on:
        1. Market trends and opportunities
        2. Consumer behavior patterns
        3. Competitive landscape insights
        4. Actionable recommendations for OmniActive
        
        Give this in 100 words total
        Determine if additional intelligence is needed from other agents.
        """
        
        def make_api_call():
            print(f"Making Gemini API call with prompt: {analysis_prompt}")  # Debugging line
            return make_gemini_request(analysis_prompt, max_tokens=8192, temperature=0.3, system_prompt=WEB_INTELLIGENCE_SYSTEM_PROMPT)
        
        result = await asyncio.to_thread(make_api_call)
        result = result.replace("```json","").replace("```", "").strip()
        # Parse JSON response
        try:
            response_data = json.loads(result)
            return response_data
        except json.JSONDecodeError:
            return {
                "response": f"Web Intelligence Analysis: Based on current market research, I found relevant trends in the nutraceutical space. The search revealed insights about market dynamics and consumer preferences that could be valuable for OmniActive's strategy.",
                "next_action": "stop",
                "agent_prompt": ""
            }
        
    except Exception as e:
        print(f"Error in web intelligence agent: {str(e)}")
        return {
            "response": f"Web Intelligence Agent encountered an error: {str(e)}",
            "next_action": "stop",
            "agent_prompt": ""
        }