import os
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import logging


def setup_groq_agent(groq_api_key):
    """
    Set up a Groq-based agent with financial and web search capabilities

    Args:
        groq_api_key: Your Groq API key
    """
    try:
        # Configure the Groq model
        groq_model = Groq(
            api_key=groq_api_key, id="llama-3.3-70b-versatile", temperature=0.7
        )

        # Create the web search agent
        web_agent = Agent(
            name="web_agent",
            model=groq_model,
            tools=[DuckDuckGo()],
            instructions=[
                "Search for relevant news and information",
                "Focus on reliable sources",
                "Provide source attribution",
            ],
            show_tool_calls=True,
            markdown=True,
        )

        # Create the financial analysis agent
        finance_agent = Agent(
            name="finance_agent",
            model=groq_model,
            tools=[
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    stock_fundamentals=True,
                )
            ],
            instructions=[
                "Analyze financial metrics and market data",
                "Present data in table format when possible",
                "Include key performance indicators",
            ],
            show_tool_calls=True,
            markdown=True,
        )

        # Create the multi-agent system
        multi_agent = Agent(
            team=[web_agent, finance_agent],
            model=groq_model,
            instructions=[
                "Combine financial data with news analysis",
                "Present information in a clear, organized format",
                "Highlight key insights and trends",
            ],
            show_tool_calls=True,
            markdown=True,
        )

        return multi_agent

    except Exception as e:
        logging.error(f"Failed to create agent: {str(e)}")
        raise


def analyze_stock(agent, symbol):
    """
    Analyze a stock using the multi-agent system

    Args:
        agent: Configured multi-agent
        symbol: Stock symbol to analyze
    """
    prompt = f"""
    Provide a comprehensive analysis of {symbol} including:
    1. Current analyst recommendations and price targets
   Present the information in a clear, organized format using tables where appropriate.
    """

    return agent.print_response(prompt, stream=True)


# Usage example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Replace with your actual Groq API key
        GROQ_API_KEY = "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"

        # Set up and run the analysis
        agent = setup_groq_agent(GROQ_API_KEY)
        analyze_stock(agent, "TSLA")

    except Exception as e:
        print(f"Error setting up or running agent: {str(e)}")
