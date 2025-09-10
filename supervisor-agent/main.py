from typing import TypedDict,Annotated,Literal
from langgraph.graph.message import add_messages,AnyMessage
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from langgraph.types import Command,interrupt
from dotenv import load_dotenv
import os
from fastapi import FastAPI

app = FastAPI()
load_dotenv()
openAI_key = os.getenv("OPENAI_API_KEY")
llm = init_chat_model("openai:gpt-4.1")

class state(BaseModel):
    user_input: str
    messages : Annotated[list[AnyMessage],add_messages]
    interrupt : str
    
class routerSchema(BaseModel):
    classification : Literal["positive","negative","neutral"] = Field(description="""when an input contains good words then 
                                 it is a positive comment,when it conatins words like bad,not good,hate it is a negative comment
                                and if it doesnt specify then it is neutral """)

llm_router = llm.with_structured_output(routerSchema)  
memory = InMemorySaver()
def positive_node(state:state):
    positive_comment = llm.invoke( state["messages"] +[
        SystemMessage(content="generate an appreciative,short and concise feedback based on the positive review"),
        HumanMessage(content=state.user_input)
        ])
    return {"messages": [AIMessage(content=positive_comment.content)]}
    
def negative_node(state:state):
    negative_comment = llm.invoke(state["messages"] +[
        SystemMessage(content="generate an apology for the review and promise to improve"),
        HumanMessage(content=state.user_input)
        ])
    return{"messages": [AIMessage(content=negative_comment.content)]}
    
def neutral_node(state:state):
       neg = llm.invoke(state["messages"] +[SystemMessage(content="generate a neutral c omment"),
        HumanMessage(content=state.user_input)
        ])
       return{ "messages" : [AIMessage(content=neg.content)]}
   
def human_approval(state: state):
    # Ask for human approval BEFORE sending response
    return interrupt({
        "question": "Do you approve this message? (approve/disapprove)",
        "generated_message": state.messages[-1].content
    })

    
    
def routerDecision(state:state):
    decision= llm_router.invoke(state.user_input)
    if decision.classification == "positive":
        return "positive_node"
    elif decision.classification == "negative":
        return "negative_node"
    else:
      return "neutral_node"

graph = StateGraph(state)
builder = graph.add_node("positive_node", positive_node)
builder = graph.add_node("negative_node", negative_node)
builder =graph.add_node("human_approval",human_approval)
builder = graph.add_node("neutral_node", neutral_node)
builder = graph.add_node("routerDecision",routerDecision)
builder = graph.add_conditional_edges(
    START,
    routerDecision,{
    "positive_node":"positive_node",
    "negative_node": "negative_node",
    "neutral_node": "neutral_node"}
                                      )
builder.add_edge("positive_node","human_approval")
builder.add_edge("human_approval", END)
builder.add_edge("negative_node",END)
builder.add_edge("neutral_node",END)


config={"configurable": {"thread_id" :"1"}}
result = builder.compile(checkpointer=memory)

@app.post("/start/")
def app_func(request :state): 
    response = result.invoke({"user_input": request.user_input,
                          "interrupt": request.user_input,
                         "messages": [HumanMessage(content=request.user_input)]
                         },
                        config)
    return {
        "agent_response": response["messages"][-1].content,
        "interrupt": response.get("__interrupt__"),
        "messages": response.get("messages")
    }

@app.post("/resume/")
def resume_feedback(request : state):
    resume_command = Command(resume=request.user_input)
    response = result.invoke(resume_command, config)
    return {
        "agent_response": response["messages"][-1].content,
        "messages": response.get("messages", [])
    }

