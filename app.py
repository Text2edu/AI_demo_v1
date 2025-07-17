import streamlit as st
import asyncio
import json
import time
import re
from datetime import datetime
from chat import chat_orchestrator
from web_intel import web_intelligence_agent
from social_intel import social_intelligence_agent
from comp_intel import competitive_intelligence_agent

# Page configuration
st.set_page_config(
    page_title="OmniActive Marketing Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        color: #e0e0e0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .agent-response {
        background: #000000;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
    }
    
    .user-message {
        background: #000000;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
        animation: pulse 2s infinite;
    }
    
    .status-inactive {
        background-color: #6c757d;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .metric-card {
        background: black;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background-color: #2a5298;
        color: black;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1e3c72;
    }
    
    /* JSON formatting styles */
    .json-output {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        max-height: 500px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
    }
    
    .stJson {
        background-color: #000000 !important;
        color: #f8f8f2 !important;
        padding: 10px !important;
        border-radius: 5px !important;
        border-left: 3px solid #17a2b8 !important;
    }
    
    /* Custom colors for JSON syntax highlighting */
    .json-key {
        color: #f92672;
    }
    
    .json-value {
        color: #a6e22e;
    }
    
    .json-string {
        color: #e6db74;
    }
    
    /* Report styling */
    .report-section {
        margin-bottom: 15px;
        padding: 10px;
        background-color: #111;
        border-radius: 5px;
    }
    
    .report-section h3 {
        margin-top: 0;
        color: #3498db;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    
    .report-section ul {
        margin-top: 5px;
        padding-left: 20px;
    }
    
    .highlight {
        color: #2ecc71;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions (moved outside of any loops)
def is_json_like(text):
    """Check if text appears to be JSON"""
    if not isinstance(text, str):
        return False
    text = str(text).strip()
    return (text.startswith('{') and text.endswith('}')) or \
           (text.startswith('[') and text.endswith(']'))

def is_report_format(text):
    """Check if the text contains multiple sections with headers"""
    if not isinstance(text, str):
        return False
    report_patterns = ["Overview:", "Sentiment Analysis:", "Actionable Insights:", 
                      "Competitor Analysis:", "Market Positioning:", "Recommendations:"]
    return any(pattern in text for pattern in report_patterns)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_history' not in st.session_state:
    st.session_state.agent_history = []
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 OmniActive Marketing Intelligence</h1>
    <p>AI-Powered Multi-Agent Market Research Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Agent Dashboard")
    
    # Agent status indicators
    st.subheader("Agent Status")
    
    agents = [
        {"name": "Chat Orchestrator", "status": "active" if not st.session_state.processing else "inactive"},
        {"name": "Web Intelligence", "status": "active" if st.session_state.current_agent == "web_intelligence_agent" else "inactive"},
        {"name": "Social Intelligence", "status": "active" if st.session_state.current_agent == "social_intelligence_agent" else "inactive"},
        {"name": "Competitive Intelligence", "status": "active" if st.session_state.current_agent == "competitive_intelligence_agent" else "inactive"}
    ]
    
    for agent in agents:
        status_class = "status-active" if agent["status"] == "active" else "status-inactive"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {status_class}"></span>
            <span>{agent['name']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    if st.button("🔍 Market Trends Analysis"):
        st.session_state.quick_query = "What are the latest trends in the nutraceutical market?"
        
    if st.button("📱 Social Sentiment Check"):
        st.session_state.quick_query = "How do customers feel about OmniActive products on social media? What are the other companies which are using omniactive in different countries"
        
    if st.button("🏆 Competitive Analysis"):
        st.session_state.quick_query = "Can u give me multiple reviews from reddit google flipkart amazon etc for Lutemax and also mention 3-4 companies which use Lutemax in their products"
        
    if st.button("📊 Product Performance"):
        st.session_state.quick_query = "How is Lutemax performing in the market?"
    
    st.divider()
    
    # Session stats
    st.subheader("📈 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", len(st.session_state.messages))
    with col2:
        st.metric("Agents Used", len(set(agent['agent'] for agent in st.session_state.agent_history)) if st.session_state.agent_history else 0)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.session_state.current_agent = None
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                content = message['content']
                
                # Display the agent name header
                agent_name = message.get('agent', 'Assistant')
                st.markdown(f"""
                <div class="agent-response" style="margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
                    <strong>{agent_name}:</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Check format and display appropriately
                if isinstance(content, str) and is_json_like(content):
                    try:
                        # Try to parse as JSON
                        json_data = json.loads(content)
                        with st.expander("Show raw JSON", expanded=False):
                            st.json(json_data)
                        
                        # Render a more readable version of the JSON
                        st.markdown('<div class="report-section">', unsafe_allow_html=True)
                        
                        # Handle different report structures
                        if "Overview" in json_data:
                            st.markdown(f"<h3>Overview</h3><p>{json_data['Overview']}</p>", unsafe_allow_html=True)
                        
                        if "Competitor Analysis" in json_data:
                            st.markdown("<h3>Competitor Analysis</h3>", unsafe_allow_html=True)
                            comp_analysis = json_data["Competitor Analysis"]
                            if isinstance(comp_analysis, dict):
                                for key, value in comp_analysis.items():
                                    st.markdown(f"<strong>{key}:</strong> {value}", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p>{comp_analysis}</p>", unsafe_allow_html=True)
                        
                        if "Companies Using OmniActive Ingredients" in json_data:
                            st.markdown("<h3>Companies Using OmniActive Ingredients</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p>{json_data['Companies Using OmniActive Ingredients']}</p>", unsafe_allow_html=True)
                        
                        if "Market Positioning Opportunities" in json_data:
                            st.markdown("<h3>Market Positioning</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p>{json_data['Market Positioning Opportunities']}</p>", unsafe_allow_html=True)
                        
                        if "Actionable Recommendations" in json_data:
                            st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
                            recs = json_data["Actionable Recommendations"]
                            if isinstance(recs, list):
                                for rec in recs:
                                    st.markdown(f"• {rec}", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p>{recs}</p>", unsafe_allow_html=True)
                        
                        # Handle other response types
                        for key, value in json_data.items():
                            if key not in ["Overview", "Competitor Analysis", "Companies Using OmniActive Ingredients", 
                                           "Market Positioning Opportunities", "Actionable Recommendations"]:
                                st.markdown(f"<h3>{key}</h3>", unsafe_allow_html=True)
                                if isinstance(value, list):
                                    for item in value:
                                        st.markdown(f"• {item}", unsafe_allow_html=True)
                                elif isinstance(value, dict):
                                    for k, v in value.items():
                                        st.markdown(f"<strong>{k}:</strong> {v}", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<p>{value}</p>", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except json.JSONDecodeError:
                        # If not valid JSON, display as regular text
                        st.markdown(f"""
                        <div class="agent-response" style="border-top-left-radius: 0; border-top-right-radius: 0;">
                            {content}
                        </div>
                        """, unsafe_allow_html=True)
                elif isinstance(content, str) and is_report_format(content):
                    # Format report-like text with better styling
                    st.markdown(f"""
                    <div class="agent-response" style="border-top-left-radius: 0; border-top-right-radius: 0;">
                        {content.replace("Overview:", "<h3>Overview</h3>")
                                .replace("Sentiment Analysis:", "<h3>Sentiment Analysis</h3>")
                                .replace("Actionable Insights:", "<h3>Actionable Insights</h3>")
                                .replace("Competitive Analysis:", "<h3>Competitive Analysis</h3>")
                                .replace("Market Positioning:", "<h3>Market Positioning</h3>")
                                .replace("Recommendations:", "<h3>Recommendations</h3>")}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Regular text response
                    st.markdown(f"""
                    <div class="agent-response" style="border-top-left-radius: 0; border-top-right-radius: 0;">
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input
    if 'quick_query' in st.session_state:
        user_input = st.session_state.quick_query
        del st.session_state.quick_query
    else:
        user_input = st.chat_input("Ask about market trends, competitor analysis, or social sentiment...")
    
    if user_input and not st.session_state.processing:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process the query
        st.session_state.processing = True
        st.rerun()

with col2:
    st.subheader("📊 Agent Activity")
    
    # Show recent agent activity
    if st.session_state.agent_history:
        st.write("Recent Agent Actions:")
        for i, activity in enumerate(st.session_state.agent_history[-5:]):
            with st.expander(f"{activity['agent']} - {activity['timestamp'][:16]}"):
                st.write(f"**Task:** {activity['task']}")
                st.write(f"**Status:** {activity['status']}")
                if activity.get('next_action'):
                    st.write(f"**Next Action:** {activity['next_action']}")
    else:
        st.info("No agent activity yet. Start a conversation to see agent actions!")

# Process user input
async def process_query(query: str):
    """Process user query through the agent chain"""
    try:
        context = " ".join([msg['content'] for msg in st.session_state.messages[-3:] if msg['role'] == 'user'])
        
        # Start with chat orchestrator
        response = await chat_orchestrator(query, context)
        
        # Add orchestrator response
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response['response'],
            'agent': 'Chat Orchestrator',
            'timestamp': datetime.now().isoformat()
        })
        
        # Track agent activity
        st.session_state.agent_history.append({
            'agent': 'Chat Orchestrator',
            'task': query,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'next_action': response['next_action']
        })
        
        # Process through agents chain
        current_response = response
        max_iterations = 3
        iteration = 0
        
        while current_response['next_action'] != 'stop' and iteration < max_iterations:
            iteration += 1
            next_agent = current_response['next_action']
            agent_prompt = current_response['agent_prompt']
            
            st.session_state.current_agent = next_agent
            
            # Call the appropriate agent
            if next_agent == 'web_intelligence_agent':
                current_response = await web_intelligence_agent(agent_prompt)
                agent_name = 'Web Intelligence Agent'
            elif next_agent == 'social_intelligence_agent':
                current_response = await social_intelligence_agent(agent_prompt)
                agent_name = 'Social Intelligence Agent'
            elif next_agent == 'competitive_intelligence_agent':
                current_response = await competitive_intelligence_agent(agent_prompt)
                agent_name = 'Competitive Intelligence Agent'
            else:
                break
            
            # Add agent response
            st.session_state.messages.append({
                'role': 'assistant',
                'content': current_response['response'],
                'agent': agent_name,
                'timestamp': datetime.now().isoformat()
            })
            
            # Track agent activity
            st.session_state.agent_history.append({
                'agent': agent_name,
                'task': agent_prompt,
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'next_action': current_response.get('next_action', 'stop')
            })
        
        st.session_state.current_agent = None
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        st.session_state.messages.append({
            'role': 'assistant',
            'content': f"I encountered an error: {str(e)}. Please try again.",
            'agent': 'System',
            'timestamp': datetime.now().isoformat()
        })
    
    finally:
        st.session_state.processing = False

# Handle processing
if st.session_state.processing and st.session_state.messages:
    with st.spinner("🤖 Processing your query through the agent network. Please wait for 3 minutes for a comprehensive report"):
        # Get the last user message
        last_user_message = None
        for msg in reversed(st.session_state.messages):
            if msg['role'] == 'user':
                last_user_message = msg['content']
                break
        
        if last_user_message:
            # Run the async function
            asyncio.run(process_query(last_user_message))
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 OmniActive Marketing Intelligence Platform | Powered by Multi-Agent AI</p>
    <p>Real-time market research • Social listening • Competitive intelligence</p>
</div>
""", unsafe_allow_html=True)