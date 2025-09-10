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

load_dotenv()
openAI_key = os.getenv("OPENAI_API_KEY")
llm = init_chat_model("openai:gpt-4.1")

class state(TypedDict):
    user_input: str
    messages : Annotated[list[AnyMessage],add_messages]
    interrupt : str
    
class routerSchema(BaseModel):
    classification : Literal["positive","negative","neutral"] = Field(description="""when an input contains good words then 
                                 it is a positive comment,when it conatins words like bad,not good,hate it is a negative comment
                                and if it doesnt specify then it is neutral """)

llm_router = llm.with_structured_output(routerSchema)  
memory = InMemorySaver()
@tool
def positive_node(state:state):
    """this node is to generate positive response for the feedback"""
    positive_comment = llm_router.invoke( state["messages"] +[
        SystemMessage(content="generate an appreciative,short and concise feedback based on the positive review"),
        HumanMessage(content=state["user_input"])
        ])
    return {"messages": [AIMessage(content=positive_comment.content)]}
    
@tool
def negative_node(state:state):
    """this is to generate an apology based on the input from user"""
    negative_comment = llm_router.invoke(state["messages"] +[
        SystemMessage(content="generate an apology for the review and promise to improve"),
        HumanMessage(content=state["user_input"])
        ])
    return{"messages": [AIMessage(content=negative_comment.content)]}
   
@tool 
def neutral_node(state:state):
    """generate a neutral response"""
    neg = llm_router.invoke(state["messages"] +[SystemMessage(content="generate a neutral c omment"),
        HumanMessage(content=state["user_input"])
        ])
    return{ "messages" : [AIMessage(content=neg.content)]}

    
tool = [positive_node,negative_node,neutral_node]
config={"configurable": {"thread_id" :"1"}}
agent = create_react_agent(
    model=llm,
    tools=tool,
    checkpointer=memory,
    prompt=""
)

while True:
    user_input = input("\nAI: Ask me anything (or type 'quit' to exit)\nUSER: ")
    
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break
    app = agent.invoke({"user_input": user_input, "interupt": "",
                         "messages": [HumanMessage(content=user_input)]
                         },
                        config)
    print(app["messages"][-1].content)

