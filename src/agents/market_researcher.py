"""Market Researcher performs market research, extracts relevant text snippet and saves results"""

from pathlib import Path
import sys
from crewai import Agent, Task
from crewai.tools import tool
import yfinance as yf
from langchain_openai import ChatOpenAI
from .base import load_prompt

tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools import financial_tools
from src.tools.memory_tools import MemoryTools

def build_market_researcher() -> Agent:
    """
    Build the market researcher agent.

    This agent performs market research, extracts relevant text snippet and stores results in vector memory. 

    Returns:
        Configured market researcher agent
    """
    # Load system prompt
    prompt = load_prompt('market_researcher.md')

    agent = Agent(
        role="Market Researcher",
        goal=f"performs market research, extracts relevant text snippet and stores results for stock. Save results to memory",
        backstory=(prompt),
        verbose=True,
        tools=[financial_tools.get_market_data, MemoryTools.save_finding],

        # ALLOW DELEGATION: Set to False because we want this agent to do the
        # research itself, not delegate to the writer (who has no tools anyway).
        allow_delegation=False,
        
       # LLM: We use temperature=0.7 for more creative, engaging writing
        # while still maintaining coherence and accuracy.
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7),

        #max_iter=15
    )

    return agent

def build_market_researcher_task(inputs: dict = None) -> Task:
    """
    Build market_researcher task:

    1 Searches markets, news, press releases, and analyst commentary
    2 Extracts relevant text snippets
    3 Saves results in vector database
   
    Returns:
        Configured market research task
    """
    ticker = inputs.get("ticker") 
    investor_mode = inputs.get("investor_mode")

    task = Task(      
    description=(
       f"Research the market landscape and sentiment for stock {ticker}.\n\n"
        f"Investment Perspective: {investor_mode}\n\n"
        "Your research must cover:\n"
        "- Fetch recent market news\n"
        "- Search the web for sentiment, risks, upcoming events\n"
        "- Extract article text snippets\n"
        "- Write brief summaries\n"
        "- Save results in vector DB\n\n"        
        f"Consider the {investor_mode.lower()} perspective when "
        "highlighting key findings.\n\n"
        "Use the available search tools to find current market information."
    ),
    expected_output=(
        "A brief summary of the market research results\n\n"
       
        "Confirm that summary saved."
    ),
    agent=build_market_researcher()    
    )

    return task