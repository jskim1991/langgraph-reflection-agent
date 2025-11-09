from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph
from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def generation_node(state: State):
    result = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [result]}

def reflection_node(state: State):
    # treat reflection as human feedback
    result = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=result.content)]}

def should_continue(state: State):
    if len(state["messages"]) > 5:
        return END
    return REFLECT

builder = StateGraph(State)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

# print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
# graph.get_graph().print_ascii()

if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(
                content="""
                    Make this tweet better:
                    @LangChainAI
                    — newly Tool Calling feature is seriously underrated.
                    After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.
                    Made a video covering their newest blog post
                """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)