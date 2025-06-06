from langchain_core.messages import HumanMessage, AIMessage


def euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate squared Euclidean distance between two points (no square root for efficiency)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return dlat ** 2 + dlon ** 2


from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def parse_langgraph_output(stream):
    results = []

    # Nếu stream là tuple, chuyển sang dict
    if isinstance(stream, tuple) and len(stream) == 2 and isinstance(stream[1], dict):
        stream = stream[1]

    for key, value in stream.items():
        if key == "supervisor":
            continue
        messages = value.get("messages", [])
        for msg in messages:
            if isinstance(msg, str):
                results.append((key, msg))
            elif isinstance(msg, AIMessage):
                results.append((key, msg.content))
    return results
