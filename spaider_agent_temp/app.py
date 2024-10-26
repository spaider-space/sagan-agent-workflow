from langchain_core.runnables.config import RunnableConfig


# LOCAL IMPORTS.
from graph import create_graph, compile_graph, print_stream


config = RunnableConfig(recursion_limit=50)
print(config)

if __name__ == "__main__":
    # creating graph workflow instance and then compiling it.
    verbose = True
    builder = create_graph()
    graph = compile_graph(builder)

    # print the mermaid diagram of the graph.
    print(graph.get_graph().draw_mermaid())
    
    while True:
        user_input = input("############# User: ")
        print_stream(graph.stream({"messages": [("user", user_input)]}, stream_mode="values", config=config))
