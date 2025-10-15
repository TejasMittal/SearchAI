import os
from dotenv import load_dotenv

load_dotenv()

from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.team import Team
from agno.tools.serpapi import SerpApiTools
from agno.tools.arxiv import ArxivTools
from agno.tools.yfinance import YFinanceTools
from agno.models.openrouter import OpenRouter
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.hackernews import HackerNewsTools
from agno.db.sqlite import SqliteDb
from uuid import uuid4
import json

app = FastAPI()

"""Models In Use:
        """

class AgentDetails(BaseModel):  # Fixed class name (PascalCase)
    user_id: int
    session_id: Optional[str] = "0"
    model: Optional[str] = "gemini-2.0-flash-001" 
    prompt: str
    best_toggle: Literal[0, 1] = 0 # 0 for false and 1 for true

class AgentResponse(BaseModel):
    response: str
    session_id: str
    user_id: int
    Links: Optional[list[dict[str, str]]]

class RephraserInput(BaseModel):
    user_prompt: str = Field(..., min_length=1, description="The prompt to be rephrased")

class RephraserOutput(BaseModel):
    original_prompt: str
    rephrased_prompt: str
    word_count: int

# Load agent configuration
with open('prompt.json', 'r', encoding='utf-8') as file:
    agent_config = json.load(file)

# Main chat model api
@app.post("/chatModel", response_model=AgentResponse)
async def response_retrieval(agent_details: AgentDetails):  # Removed Path()
    try:
        db = SqliteDb(db_file="localdb/searchAI.db")

        # Fix session_id handling
        session_id: str
        if agent_details.session_id != "0":
            session_id = agent_details.session_id
        else:
            session_id = str(uuid4())

        # Create agent
        if agent_details.best_toggle == 0:
            agent = Agent(model=Gemini(id=agent_details.model, max_output_tokens=5000, temperature=0.5, top_p=0.3), 
            name="SearchAI", instructions= agent_config["searchAI"]["instructions"], description = agent_config["searchAI"]["description"], db=db, session_id=session_id, user_id=agent_details.user_id, enable_agentic_memory=True, enable_user_memories=True,read_chat_history=True, num_history_runs=5, add_history_to_context=True, markdown=True, tools=[SerpApiTools(api_key=os.getenv("SERPAPI_API_KEY")), ArxivTools(), YFinanceTools()], debug_mode=True)

        elif agent_details.best_toggle == 1:
            # **Change logic try making different task agents.**
            reddit_researcher = Agent(
                name="DuckDuckGoAI",
                model=Gemini(id="gemini-2.0-flash-001"),
                tools=[DuckDuckGoTools()],
                add_name_to_context=True, instructions= agent_config["DuckDuckGoAI"]["instructions"], description = agent_config["DuckDuckGoAI"]["description"]
            )

            hackernews_researcher = Agent(
                name="HackerNewsAI",
                model=Gemini(id="gemini-2.0-flash-001"),
                role="Research a topic on HackerNews.",
                tools=[HackerNewsTools()],
                add_name_to_context=True, instructions= agent_config["HackerNewsAI"]["instructions"], description = agent_config["HackerNewsAI"]["description"]
            )

            academic_paper_researcher = Agent(
                name="GoogleArxivAI",
                model=Gemini(id="gemini-2.0-flash-001"),
                role="Research academic papers and scholarly content",
                tools=[GoogleSearchTools(), ArxivTools()],
                add_name_to_context=True, instructions= agent_config["GoogleArxivAI"]["instructions"], description = agent_config["GoogleArxivAI"]["description"]
            )

            twitter_researcher = Agent(
                name="YFinanceAI",
                model=Gemini(id="gemini-2.0-flash-001"),
                role="Research trending discussions and real-time updates",
                tools=[YFinanceTools()],
                add_name_to_context=True, instructions= agent_config["YFinanceAI"]["instructions"], description = agent_config["YFinanceAI"]["description"]
            )

            SerpAPIAgent = Agent(
                name="SerpAPIAgent",
                model=Gemini(id="gemini-2.0-flash-001"),
                role="Research trending discussions and real-time updates",
                tools=[SerpApiTools(api_key=os.getenv("SERPAPI_API_KEY"))],
                add_name_to_context=True, instructions= agent_config["SerpAPIAgent"]["instructions"], description = agent_config["SerpAPIAgent"]["description"]
            )

            agent = Team(
                name="SelectorAI",
                model=Gemini(id="gemini-2.0-flash-001"),
                members=[
                    reddit_researcher,
                    hackernews_researcher,
                    academic_paper_researcher,
                    twitter_researcher,
                    SerpAPIAgent
                ],
                db=db,
                add_history_to_context=True,
                num_history_runs=3,
                add_session_state_to_context=True,  # Required so the agent is aware of the session state
                enable_agentic_state=True,
                enable_user_memories=True,
                instructions= agent_config["SelectorAI"]["instructions"], description = agent_config["SelectorAI"]["description"],
                delegate_task_to_all_members=True,
                markdown=True,
                debug_mode=True
            )
         
        # Run the agent
        response = await agent.arun(agent_details.prompt)
        
        # Extract the actual response content
        response_content = ""
        if hasattr(response, 'content'):
            response_content = response.content

        print(response_content)

        # defining variables:
        final_content = []
        Links:list[dict[str, str]] = []

        #checking whether the souces are present or not:
        if response_content.__contains__("Sources and references:"):
            final_content = response_content.split("Sources and references:")
        elif response_content.__contains__("📚 Sources"):
            final_content = response_content.split("📚 Sources")
        else:
            final_content.append(response_content)

        print(final_content)

        # if sources present:
        if len(final_content) > 1:

            print(final_content[0], final_content[1])

            if len(final_content[1]) > 0:
            
                sources = []
                if ") | [" in final_content[1]:
                    sources = final_content[1].split(") | [")
                else:
                    sources.append(final_content[1])
            
                if len(sources) > 0:
                    for source in sources:
                        if "(" in source:
                            content = source.split("(")

                            title = ""
                            link = ""
                            if "[" in content[0]:
                                title = content[0].replace("[", "").strip()
                                title = title.replace("]", "").strip()
                            elif "]" in content[0]:
                                title = content[0].replace("]", "").strip()
                            
                            if ")" in content[1]:
                                link = content[1].replace(")", "").strip()
                            else:
                                link = content[1].strip()
                            

                            Links.append({"title":title, "url":link})

            print(Links)
            print(response_content)
            print(final_content[0])

        return AgentResponse(
            response= final_content[0],
            session_id=session_id,
            user_id=agent_details.user_id,
            Links = Links
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="prompt.json file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in prompt.json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/prompt-rephraser-v3")
async def rephraser_v3(userInput: RephraserInput):
    """
    Version 3: Uses a direct prompt engineering approach
    """
    
    # Minimal, direct instruction
    system_message = "You are a prompt optimization assistant. Rephrase user prompts to be clear, specific, and under 50 words. Output only the rephrased prompt with no additional text."
    
    try:
        rephraserAgent = Agent(
            model=Gemini(
                id="gemini-2.0-flash-exp",
                max_output_tokens=100,
                temperature=0.1,  # Lowest temperature
            ),
            system_message=system_message,
            markdown=False,
        )
        
        # Use a constraint-based prompt
        user_message = f"""Original: {userInput.user_prompt}
                            Task: Rewrite the above as a clear, specific prompt for an LLM (maximum 50 words).
                            Output: [write only the rephrased prompt here]"""
        
        response = rephraserAgent.run(user_message)
        rephrased = response.content.strip()
        
        # Extract content after "Output:" if present
        if "output:" in rephrased.lower():
            rephrased = rephrased.split(":", 1)[1].strip()
        
        # Clean brackets if present
        rephrased = rephrased.strip('[]')
        
        word_count = len(rephrased.split())
        
        return {
            "original_prompt": userInput.user_prompt,
            "rephrased_prompt": rephrased,
            "word_count": word_count,
            "version": "v3"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add a test endpoint to verify the API is working
@app.get("/health")
def health_check():
    return {"status": "healthy"}