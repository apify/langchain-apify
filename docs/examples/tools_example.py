import os

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_apify import ApifyActorsTool

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
os.environ['APIFY_API_TOKEN'] = 'YOUR_APIFY_API_TOKEN'

model = ChatOpenAI(model='gpt-4o-mini')

tool = ApifyActorsTool(actor_id='apify/rag-web-browser')

# Example: Use the tool directly to call the Apify Actor
print('Calling tool ...')
results = tool.invoke(input={'run_input': {'query': 'what are Apify Actors?', 'maxResults': 3}})
for result in results:
    print(result)

# Example: Use the tool with an agent
tools = [tool]
agent = create_react_agent(model, tools)

for chunk in agent.stream(
    {'messages': [('human', 'search for what is Apify?')]},
    stream_mode='values',
):
    msg = chunk['messages'][-1]
    # skip tool messages
    if isinstance(msg, ToolMessage):
        continue
    msg.pretty_print()
