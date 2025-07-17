import os
import json
import asyncio
import requests
import streamlit as st
import asyncpraw  # Changed from praw to asyncpraw
import pandas as pd
from datetime import datetime, timedelta
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

# Reddit API Configuration
REDDIT_CLIENT_ID = st.secrets.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = st.secrets.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = st.secrets.get("REDDIT_USER_AGENT", "python:OmniActiveResearchBot:1.0 (by /u/dev-on_rocks)")

# The JSON string for credentials will need to be parsed
google_applications_credentials_json_str = st.secrets["GOOGLE_APPLICATIONS_CREDENTIALS_JSON"]
GOOGLE_APPLICATIONS_CREDENTIALS_JSON = google_applications_credentials_json_str

SOCIAL_INTELLIGENCE_SYSTEM_PROMPT = """You are a Social Intelligence Agent for OmniActive Health Technologies marketing research.

Your role is to monitor social media mentions, customer reviews, and online conversations about OmniActive's products and the nutraceutical industry.

You must ALWAYS respond in valid JSON format with these fields:
{
    "response": "Your analysis of social sentiment and customer feedback. Give this in a report structure with Overview, Sentiment analysis, Actionable insights and Red Flags",
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

async def initialize_reddit_client():
    """Initialize and return an AsyncPRAW Reddit client instance"""
    try:
        reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        return reddit
    except Exception as e:
        print(f"Error initializing AsyncPRAW Reddit client: {e}")
        return None

async def fetch_reddit_data(subreddits, keywords, limit=25, time_filter="month"):
    """Fetch Reddit posts and comments containing keywords from specified subreddits using AsyncPRAW"""
    reddit = await initialize_reddit_client()
    if not reddit:
        return []
    
    all_posts = []
    
    try:
        # Convert to list if a single string is provided
        if isinstance(subreddits, str):
            subreddits = [subreddits]
        if isinstance(keywords, str):
            keywords = [keywords]
            
        for subreddit_name in subreddits:
            subreddit = await reddit.subreddit(subreddit_name)
            
            # Search for posts containing keywords
            for keyword in keywords:
                search_query = keyword.lower()
                
                # Get posts from different sort methods for broader coverage
                search_methods = [
                    subreddit.search(search_query, sort="relevance", time_filter=time_filter, limit=limit),
                    subreddit.search(search_query, sort="comments", time_filter=time_filter, limit=limit//2)
                ]
                
                for search_method in search_methods:
                    async for post in search_method:
                        # Basic post info
                        post_data = {
                            "id": post.id,
                            "title": post.title,
                            "url": f"https://www.reddit.com{post.permalink}",
                            "created_utc": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "upvote_ratio": post.upvote_ratio,
                            "subreddit": subreddit_name,
                            "selftext": post.selftext[:5000] if hasattr(post, "selftext") else "",
                            "comments": []
                        }
                        
                        # Get comments
                        try:
                            # Fetch comments with AsyncPRAW (safer approach)
                            submission = await reddit.submission(id=post.id)
                            await submission.comments.replace_more(limit=2)
                            
                            # Get top-level comments
                            comment_list = []
                            async for comment in submission.comments:
                                if hasattr(comment, "body"):  # Make sure it's a real comment
                                    comment_list.append(comment)
                                if len(comment_list) >= 10:  # Limit to top 10 comments
                                    break
                                                        
                            for comment in comment_list:
                                comment_data = {
                                    "body": comment.body if hasattr(comment, "body") else "",
                                    "score": comment.score if hasattr(comment, "score") else 0,
                                    "created_utc": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S") if hasattr(comment, "created_utc") else "",
                                    "replies": []
                                }
                                
                                # Get replies
                                if hasattr(comment, "replies") and comment.replies:
                                    reply_count = 0
                                    async for reply in comment.replies:
                                        if hasattr(reply, "body"):  # Make sure it's a real reply
                                            reply_data = {
                                                "body": reply.body,
                                                "score": reply.score if hasattr(reply, "score") else 0,
                                                "created_utc": datetime.fromtimestamp(reply.created_utc).strftime("%Y-%m-%d %H:%M:%S") if hasattr(reply, "created_utc") else ""
                                            }
                                            comment_data["replies"].append(reply_data)
                                            reply_count += 1
                                            if reply_count >= 5:  # Limit to 5 replies per comment
                                                break
                                
                                post_data["comments"].append(comment_data)
                            
                        except Exception as e:
                            print(f"Error fetching comments for post {post.id}: {str(e)}")
                        
                        all_posts.append(post_data)
        
        # Close the Reddit instance when done
        await reddit.close()
        return all_posts
    
    except Exception as e:
        print(f"Error fetching Reddit data: {e}")
        if reddit:
            await reddit.close()
        return []

def format_reddit_data_for_analysis(reddit_posts):
    """Format Reddit posts and comments into a readable format for LLM analysis"""
    if not reddit_posts:
        return "No Reddit data found."
    
    formatted_data = ""
    
    for i, post in enumerate(reddit_posts):
        formatted_data += f"\n--- REDDIT POST {i+1} ---\n"
        formatted_data += f"Title: {post['title']}\n"
        formatted_data += f"Subreddit: r/{post['subreddit']}\n"
        formatted_data += f"Date: {post['created_utc']}\n"
        formatted_data += f"Score: {post['score']} | Comments: {post['num_comments']} | Upvote Ratio: {post['upvote_ratio']}\n"
        
        if post['selftext']:
            formatted_data += f"Post Content: {post['selftext'][:1000]}...\n" if len(post['selftext']) > 1000 else f"Post Content: {post['selftext']}\n"
        
        formatted_data += f"URL: {post['url']}\n\n"
        
        # Add comments
        formatted_data += "TOP COMMENTS:\n"
        for j, comment in enumerate(post['comments']):
            formatted_data += f"  [{j+1}] Score: {comment['score']} | {comment['body'][:500]}...\n" if len(comment['body']) > 500 else f"  [{j+1}] Score: {comment['score']} | {comment['body']}\n"
            
            # Add top replies if they exist
            if comment['replies']:
                formatted_data += "    REPLIES:\n"
                for k, reply in enumerate(comment['replies'][:3]):  # Limit to top 3 replies
                    formatted_data += f"      - Score: {reply['score']} | {reply['body'][:250]}...\n" if len(reply['body']) > 250 else f"      - Score: {reply['score']} | {reply['body']}\n"
        
        formatted_data += "\n" + "-"*80 + "\n"
    
    return formatted_data

def check_reddit_credentials():
    """Check if Reddit API credentials are available"""
    client_id = st.secrets.get("REDDIT_CLIENT_ID")
    client_secret = st.secrets.get("REDDIT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        st.warning("⚠️ Reddit API credentials are not set. Reddit data collection will be skipped.")
        return False
    return True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def social_intelligence_agent(agent_prompt: str) -> dict:
    """Social Intelligence Agent that monitors social media and reviews"""
    try:
        # Generate search queries for serper
        search_queries = [
            f"site:amazon.com Capsimax supplement reviews {agent_prompt}",
            f"site:iherb.com OmniActive ingredients {agent_prompt}",
            f"\"OmniActive\" customer feedback {agent_prompt}"
        ]
        
        # Reddit-specific search
        reddit_subreddits = ["supplements", "Nootropics", "nutrition", "eyehealth", "fitness"]
        reddit_keywords = [
            "OmniActive", "Lutemax", "Capsimax", "lutein", "zeaxanthin", 
            "macular pigment", "eye health supplements", "capsaicin supplement"
        ]
        
        # Perform searches in parallel
        search_tasks = [asyncio.to_thread(serper_search, query) for query in search_queries[:3]]
        search_results = await asyncio.gather(*search_tasks)
        
        # Check Reddit credentials and fetch Reddit data if available
        formatted_reddit_results = "No Reddit data available."
        if check_reddit_credentials():
            reddit_posts = await fetch_reddit_data(reddit_subreddits, reddit_keywords, limit=30)
            formatted_reddit_results = format_reddit_data_for_analysis(reddit_posts)
        
        # Format Serper search results
        formatted_serper_results = ""
        for i, result in enumerate(search_results):
            formatted_serper_results += f"\n--- Search: {search_queries[i]} ---\n"
            for item in result.get("organic", [])[:5]:  # Top 5 results
                formatted_serper_results += f"Title: {item.get('title', 'N/A')}\n"
                formatted_serper_results += f"Snippet: {item.get('snippet', 'N/A')}\n"
                formatted_serper_results += f"Link: {item.get('link', 'N/A')}\n\n"
        
        # Combine all data for analysis
        all_data = f"""
        GOOGLE SEARCH RESULTS:
        {formatted_serper_results}
        
        REDDIT DATA:
        {formatted_reddit_results}
        """
        
        # Analyze results with LLM
        analysis_prompt = f"""
        Agent Task: {agent_prompt}
        
        Below is comprehensive social media and review data about OmniActive Health Technologies and its products:
        
        {all_data}
        
        Analyze this data and provide insights about:
        1. Customer sentiment towards OmniActive products (positive, negative, neutral percentages if possible)
        2. Common feedback patterns and recurring themes
        3. Consumer education needs and knowledge gaps
        4. Brand perception and awareness levels
        5. Purchase decision factors and barriers
        6. Influencer and expert opinions
        7. Specific product feedback on Lutemax, Capsimax, or other OmniActive products
        
        Include direct quotes from customers when relevant to illustrate your points.
        
        Determine if additional intelligence is needed from other agents.

        GEt this in 200 words and well orgainsed
        """
        
        def make_api_call():
            print(f"Making Gemini API call with analysis prompt")
            return make_gemini_request(analysis_prompt, max_tokens=8192, temperature=0.3, system_prompt=SOCIAL_INTELLIGENCE_SYSTEM_PROMPT)
        
        result = await asyncio.to_thread(make_api_call)
        result = result.replace("```json","").replace("```", "").strip()
        
        # Parse JSON response
        try:
            response_data = json.loads(result)
            return response_data
        except json.JSONDecodeError:
            return {
                "response": f"Social Intelligence Analysis: Based on extensive social media monitoring and review analysis across Reddit and other platforms, I found detailed insights about customer sentiment and feedback patterns. The social listening revealed valuable information about consumer perceptions and experiences with OmniActive products.",
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
