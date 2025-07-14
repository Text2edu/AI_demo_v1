
import os
import json
import asyncio
import requests
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import streamlit as st
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


CHAT_SYSTEM_PROMPT = """You are a Marketing Intelligence Chat Orchestrator for OmniActive Health Technologies. 

Your role is to analyze user queries and decide which agent should handle the request, or if you should provide a direct response.

You must ALWAYS respond in valid JSON format with these fields:
{
    "response": "Your response to the user",
    "next_action": "web_intelligence_agent" | "social_intelligence_agent" | "competitive_intelligence_agent" | "stop",
    "agent_prompt": "Specific instructions for the agent (if next_action is not 'stop')"
}

Available Agents:
1. "web_intelligence_agent" - For market trends, news, research papers, industry analysis
2. "social_intelligence_agent" - For social media mentions, reviews, customer sentiment
3. "competitive_intelligence_agent" - For competitor analysis, pricing, product comparisons

Use "stop" when:
- You can answer directly without needing search
- The conversation is complete
- The user is just chatting

Examples:
- "What are the latest trends in nutraceuticals?" → web_intelligence_agent
- "How do customers feel about Lutemax?" → social_intelligence_agent  
- "What is DSM doing in the lutein market?" → competitive_intelligence_agent
- "Hello, how are you?" → stop

Always be helpful and professional. Focus on OmniActive's products like Lutemax, Capsimax, and other nutraceutical ingredients."""

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
    """Make a request to Gemini API with proper system prompt handling"""
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def chat_orchestrator(prompt: str, context: Optional[str] = None) -> dict:
    """Main chat orchestrator that decides which agent to use"""
    try:
        if context:
            full_prompt = f"Context: {context}\n\nUser Query: {prompt}"
        else:
            full_prompt = prompt
        
        def make_api_call():
            return make_gemini_request(full_prompt, max_tokens=1024, temperature=0.3, system_prompt=CHAT_SYSTEM_PROMPT)
        
        result = await asyncio.to_thread(make_api_call)
        result = result.replace("```json","").replace("```", "").strip()
        # Parse JSON response
        try:
            response_data = json.loads(result)
            print(f"Chat Orchestrator Response: {response_data}")  # Debugging line
            return response_data
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "response": "I'm here to help with marketing intelligence queries. What would you like to know about OmniActive's products or the market?",
                "next_action": "stop",
                "agent_prompt": ""
            }
        
    except Exception as e:
        print(f"Error in chat orchestrator: {str(e)}")
        return {
            "response": f"I encountered an error: {str(e)}. Please try again.",
            "next_action": "stop",
            "agent_prompt": ""
        }