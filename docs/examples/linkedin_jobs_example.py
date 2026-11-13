import os

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from langchain_apify import ApifyActorsTool, ApifyWrapper

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
os.environ['APIFY_TOKEN'] = 'YOUR_APIFY_TOKEN'

model = ChatOpenAI(model='gpt-5.4-mini')

ACTOR_ID = 'johnvc/linkedin-jobs-api'


# Each dataset item is one job: title, companyName, location, seniorityLevel, employmentType,
# postedDate, jobUrl and description, plus salary, salaryMin and salaryMax when the listing states a range.
def job_to_document(item: dict) -> Document:
    title = item.get('title', '')
    company = item.get('companyName', '')
    description = item.get('description') or ''
    return Document(
        page_content=f'{title} at {company}\n\n{description}',
        metadata={
            'source': item.get('jobUrl'),
            'company': item.get('companyName'),
            'location': item.get('location'),
            'posted_date': item.get('postedDate'),
            'salary': item.get('salary'),
        },
    )


# Example: Run the Actor once with a typed input and load the jobs as Documents.
# maxJobs caps the number of results, and so the cost of the run.
print('Calling Actor ...')
apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id=ACTOR_ID,
    run_input={
        'keyword': 'machine learning engineer',
        'location': 'United States',
        'timeRange': 'past-week',
        'maxJobs': 10,
    },
    dataset_mapping_function=job_to_document,
)
documents = loader.load()
for doc in documents:
    print(doc.page_content.splitlines()[0], '|', doc.metadata['location'], '|', doc.metadata['salary'] or 'no salary')

# Example: Use the same Actor as a tool with an agent.
# The tool's argument schema comes from the Actor's input schema, so the agent
# sees keyword, location, timeRange and maxJobs as named fields.
tool = ApifyActorsTool(actor_id=ACTOR_ID)
agent = create_agent(model, [tool])

for chunk in agent.stream(
    {
        'messages': [
            (
                'human',
                'Which companies posted machine learning engineer jobs in the United States this week, '
                'and what salary ranges do they list? Look at 25 jobs at most.',
            )
        ]
    },
    stream_mode='values',
):
    msg = chunk['messages'][-1]
    # skip tool messages
    if isinstance(msg, ToolMessage):
        continue
    msg.pretty_print()
