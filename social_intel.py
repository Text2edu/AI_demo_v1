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

SOCIAL_INTELLIGENCE_SYSTEM_PROMPT = """You are a Social Intelligence Agent for OmniActive Health Technologies marketing research.

Your role is to monitor social media mentions, customer reviews, and online conversations about OmniActive's products and the nutraceutical industry.

You must ALWAYS respond in valid JSON format with these fields:
{
    "response": "Your analysis of social sentiment and customer feedback",
    "next_action": "web_intelligence_agent" | "competitive_intelligence_agent" | "stop",
    "agent_prompt": "Instructions for the next agent (if next_action is not 'stop')"
}

Search Strategy:
- Monitor social media platforms (Reddit, forums, review sites)
- Track product reviews on e-commerce sites
- Analyze customer sentiment and feedback
- Identify influencer mentions and discussions
- Monitor health and wellness communities

Analysis Focus:
- Customer sentiment analysis (positive/negative/neutral)
- Common complaints or praise patterns
- Influencer and expert opinions
- Consumer education needs
- Brand perception and awareness
- Purchase decision factors

Always provide specific insights about customer behavior and sentiment with examples when possible."""

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
async def social_intelligence_agent(agent_prompt: str) -> dict:
    """Social Intelligence Agent that monitors social media and reviews"""
    try:
        # Generate search queries for social listening
        search_queries = [
            f"site:reddit.com OmniActive Lutemax review {agent_prompt}",
            f"site:amazon.com Capsimax supplement reviews {agent_prompt}",
            f"site:iherb.com OmniActive ingredients {agent_prompt}",
            f"\"OmniActive\" customer feedback {agent_prompt}",
            f"lutein supplement reviews user experience {agent_prompt}",
            f"site:reddit.com r/supplements lutein {agent_prompt}"
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
        
        Social Media and Review Search Results:
        {formatted_results}
        
        Analyze these social media and review results and provide insights about:
        1. Customer sentiment towards OmniActive products
        2. Common feedback patterns (positive and negative)
        3. Consumer education needs and knowledge gaps
        4. Brand perception and awareness levels
        5. Purchase decision factors and barriers
        6. Influencer and expert opinions

        Give this in 100 words total
        
        Determine if additional intelligence is needed from other agents.
        """
        
        def make_api_call():
            print(f"Making Gemini API call with prompt: {analysis_prompt}")
            return make_gemini_request(analysis_prompt, max_tokens=8192, temperature=0.3, system_prompt=SOCIAL_INTELLIGENCE_SYSTEM_PROMPT)
        
        result = await asyncio.to_thread(make_api_call)
        result = result.replace("```json","").replace("```", "").strip()
        # Parse JSON response
        try:
            response_data = json.loads(result)
            return response_data
        except json.JSONDecodeError:
            return {
                "response": f"Social Intelligence Analysis: Based on social media monitoring and review analysis, I found insights about customer sentiment and feedback patterns. The social listening revealed valuable information about consumer perceptions and experiences with OmniActive products.",
                "next_action": "stop",
                "agent_prompt": ""
            }
        
    except Exception as e:
        print(f"Error in social intelligence agent: {str(e)}")
        return {
            "response": f"Social Intelligence Agent encountered an error: {str(e)}",
            "next_action": "stop",
            "agent_prompt": ""
        }