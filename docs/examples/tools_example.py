import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_apify import ApifyActorsTool

if "OPENAI_API_KEY" not in os.environ:
    msg = "OPENAI_API_KEY environment variable is not set."
    raise ValueError(msg)

if "APIFY_API_TOKEN" not in os.environ:
    msg = "APIFY_API_TOKEN environment variable is not set."
    raise ValueError(msg)

model = ChatOpenAI(model="gpt-4o-mini")

tool = ApifyActorsTool(actor_id="apify/rag-web-browser")

# Use the tool directly to call the Apify Actor
# results: list[dict] = tool.invoke(
#    input={"run_input": {"query": "what is Apify?", "maxResults": 3}}
# )
# for result in results:
#    print(result)

# Use the tool with an agent
tools = [tool]
agent = create_react_agent(model, tools)

for chunk in agent.stream(
    {"messages": [("human", "search for what is Apify?")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
