async def chart_agent(prompt: str) -> dict:
    """Handle chart generation requests"""
    try:
        # Here we would implement the logic to generate charts based on the prompt
        # For now, let's return a placeholder response
        chart_data = {
            "type": "line",
            "data": [1, 2, 3, 4, 5],
            "labels": ["Jan", "Feb", "Mar", "Apr", "May"]
        }
        return {
            "response": f"Generated chart data for prompt: {prompt}",
            "chart_data": chart_data,
            "next_action": "stop",
            "agent_prompt": ""
        }
    except Exception as e:
        return {
            "response": f"Error generating chart: {str(e)}",
            "next_action": "stop",
            "agent_prompt": ""
        }