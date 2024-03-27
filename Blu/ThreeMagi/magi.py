from typing import Any, Dict, List
from typing import Optional
import Blu.Utils as Utils
from Blu.Utils.Utils import readMarkdownFile
import Blu.OpenAI.GPTAssistant as GPTAssistant
from openai import OpenAI
from anthropic import Anthropic
import openai
import time
import json

client = OpenAI()

tools = GPTAssistant.generateToolsConfig(GPTAssistant.exampleFunction)
toolsDict = json.loads(tools)

print(json.dumps(toolsDict, indent=4))

instructions: str = readMarkdownFile("../markdown/personalities/melchior.md")

problemPrompt: str = readMarkdownFile("../markdown/testPrompts/decisionProblems/decisionProblem1.md")

staticTools = [toolsDict]

runSuccessful: bool = False

while not runSuccessful:
    assistant = client.beta.assistants.create(
            name="Balthazar",
            instructions=instructions,
            tools=staticTools,
            model="gpt-4-1106-preview")

    thread: openai.beta.threads = client.beta.threads.create()

    message: openai.beta.threads.messages = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=problemPrompt)


    run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
            )

    while run.status != "requires_action" and run.status != "failed":
        run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
                )
        time.sleep(1)
        print("waiting")
        print(run.status)

    runSuccessful = run.status != "failed"

print(run.required_action)

run = client.beta.threads.runs.submit_tool_outputs(
  thread_id=thread.id,
  run_id=run.id,
  tool_outputs=[
      {
        "tool_call_id": run.required_action.submit_tool_outputs.tool_calls[0].id,
        "output": True,
      },
    ]
)

run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
            )
time.sleep(1)
print(run.status)

while run.status != "requires_action" and run.status != "completed":
    run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
            )
    time.sleep(1)
    print("waiting")
    print(run.status)

messages: List[openai.beta.threads.messages] = client.beta.threads.messages.list(
  thread_id=thread.id
)

print(len(list(messages)))

for m in messages:
    print()
    print(m)
