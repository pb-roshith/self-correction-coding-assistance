from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
import subprocess
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    task : str
    code : str
    error : Optional[str]
    iterations : int
    max_iterations : int

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def programmer_node(state: AgentState):
    '''Generates or fixes code based on the current state.'''
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Python programmer. Write only raw python code without markdown blocks."),
        ("user", "Task: {task}\n\nPrevious Error: {error}\n\nHelpful Hint: If there is an error, fix the logic. Return ONLY the code.")
    ])

    chain = prompt | model

    response = chain.invoke({"task": state['task'], "error":state['error'] or None})

    clean_code = response.content.replace("```python", "").replace("```", "").strip()

    return {"code": clean_code, "iterations": state["iterations"] + 1}

def executor_node(state: AgentState):
    '''Executes the code and captures errors.'''
    try:
        result = subprocess.run(
            ["python", "-c", state['code']], 
            capture_output=True, 
            text=True, 
            timeout=5
        )

        if result.returncode == 0:
            print("EXECUTION SUCCESS")
            return {"error": None}
        else:
            print(f"EXECUTION FAILED (Attempt {state['iterations']})")

    except Exception as e:
        return {"error": str(e)}
    
def should_continue(state: AgentState):
    '''Determines if we should loop back or stop.'''
    if state['error'] is None:
        return "end"
    if state['iterations'] >= state['max_iterations']:
        return "end"
    return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("programmer", programmer_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("programmer")
workflow.add_edge("programmer", "executor")

workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue": "programmer",
        "end": END
    }
)

app = workflow.compile()

inputs = {
    "task": "Write a script that creates a list of numbers 1-10, put even numbers in a seperate list and odd numbers in a seperate list, finally print the both",
    "code": "",
    "error": None,
    "iterations": 0,
    "max_iterations": 3
}

for event in app.stream(inputs):
    print(event)