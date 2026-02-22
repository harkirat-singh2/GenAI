from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Annotated
from langchain_core.messages import HumanMessage, ToolMessage
import requests
from dotenv import load_dotenv

load_dotenv()

