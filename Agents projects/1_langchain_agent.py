from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

