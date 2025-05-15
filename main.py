import logging
import time

import streamlit as st

from langchain_core.messages import HumanMessage

from agents import supervisor

# Cấu hình giao diện
st.set_page_config(page_title="30Shine Chatbot", page_icon="💬")
st.title("💇‍♂️ 30Shine Chatbot Assistant")
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello anh, em là Janie trợ lý cá nhân của anh tại 30Shine. Em có thể giúp gì cho anh nè?"}
    ]

# Display all previous messages
for message in st.session_state.messages:
    print("message", message)
    with st.chat_message(message["role"]):
        if message["content"]:
            st.write(message["content"])

# Giao diện nhập tin nhắn
if prompt := st.chat_input("Your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

        # Generate and display assistant response
    with st.chat_message("assistant"):
        start_time = time.time()
        logging.info("Generating response...")
        with st.spinner("Processing..."):
            inputs = {
                "messages": [
                    HumanMessage(
                        content=prompt
                    )
                ],
            }

    last_msg = None

    config = {"configurable": {"thread_id": "1", "recursion_limit": 5}}
    for i, chunk in supervisor.stream(inputs, subgraphs=True, config=config):
        messages = chunk.get('agent', {}).get('messages', [])
        if messages:
            last_msg = messages[-1].content

        # Thêm phản hồi vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": last_msg})
    with st.chat_message("assistant"):
        st.write(last_msg)
