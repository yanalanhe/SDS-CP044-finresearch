"""The Financial Crew represents group of agents working together to achieve a set of financial service tasks:
    1. Manager Agent delegates tasks to financial analyst and market researcher and produces final report
    2. Finacial Analyst Agent analyses company-specific financial data
    3. Market Researcher Agent searches markets, extracts relevant text snippets and stores results memory"""

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from agents.base import load_prompt
from agents.manager import build_manager
from agents.financial_analyst import build_financial_analyst, build_financial_analyst_task
from agents.market_researcher import build_market_researcher, build_market_researcher_task


def build_financial_crew(inputs: dict = None) -> Crew:
    """
    Build the financial crew.

    The Financial Crew represents group of agents working together to achieve a set of financial service tasks:
    1. Manager Agent delegates tasks to financial analyst and market researcher and produces final report
    2. Finacial Analyst Agent analyses company-specific financial data
    3. Market Researcher Agent searches markets, extracts relevant text snippets and stores results memory                         
    """ 
    # Validate inputs
    if inputs is None:
        raise ValueError("Inputs are required to build the crew")
    
    ticker = inputs.get("ticker")
    if not ticker:
        raise ValueError("Ticker symbol is required in inputs")
    
    print(f"🔨  Building crew for {ticker}...")

    # Build agents
    financial_analyst = build_financial_analyst()
    market_researcher = build_market_researcher()
       
    # Build tasks
    financial_analyst_task = build_financial_analyst_task(inputs)
    market_researcher_task = build_market_researcher_task(inputs)
    
    crew = Crew(  
        agents=[financial_analyst, market_researcher],
        tasks=[financial_analyst_task, market_researcher_task],

        # The process hierarchical is used for the crew since the introduced manager agent is used 
        # to coorrinate tasks delegated to the financial analyst agent and market research agent. 
        # Both the agents work in parallel for their own assigned tasks.
        # Technically, as specified in CewAI framework, when a manager agent is used for a crew,
        # the process must be set as hierarchical. 
        process=Process.hierarchical,

        manager_agent=build_manager(),
        verbose=True           
    )

    return crew