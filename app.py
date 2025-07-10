import streamlit as st
import asyncio
import json
import time
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
</style>
""", unsafe_allow_html=True)

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
        st.session_state.quick_query = "How do customers feel about OmniActive products on social media?"
        
    if st.button("🏆 Competitive Analysis"):
        st.session_state.quick_query = "What are our main competitors doing in the lutein market?"
        
    if st.button("📊 Product Performance"):
        st.session_state.quick_query = "How is Lutemax performing in the market?"
    
    st.divider()
    
    # Session stats
    st.subheader("📈 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", len(st.session_state.messages))
    with col2:
        st.metric("Agents Used", len(set(agent['agent'] for agent in st.session_state.agent_history)))
    
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
                st.markdown(f"""
                <div class="agent-response">
                    <strong>{message.get('agent', 'Assistant')}:</strong> {message['content']}
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
                'next_action': current_response['next_action']
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
    with st.spinner("🤖 Processing your query through the agent network..."):
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