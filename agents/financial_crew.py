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


def build_financial_crew() -> Crew:
    """
    Build the financial crew.

    The Financial Crew represents group of agents working together to achieve a set of financial service tasks:
    1. Manager Agent delegates tasks to financial analyst and market researcher and produces final report
    2. Finacial Analyst Agent analyses company-specific financial data
    3. Market Researcher Agent searches markets, extracts relevant text snippets and stores results memory                         
    """    

    crew = Crew(  
        agents=[build_financial_analyst(), build_market_researcher()],
        tasks=[build_financial_analyst_task(), build_market_researcher_task()],

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