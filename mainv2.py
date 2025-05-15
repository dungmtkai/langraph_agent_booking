import logging
import pprint
import time

import streamlit as st
from datetime import datetime

from langchain_core.messages import HumanMessage

from agentsv3 import app
from utils import pretty_print_messages, parse_langgraph_output  # bạn có thể sửa lại để trả text thay vì print
from agents import supervisor  # thay bằng module thật sự của bạn

# Cấu hình giao diện
st.set_page_config(page_title="30Shine Chatbot", page_icon="💬")
st.title("💇‍♂️ 30Shine Chatbot Assistant")
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello anh, em là Janie trợ lý cá nhân của anh tại 30Shine. Em có thể giúp gì cho anh nè?"}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    config = {"configurable": {"thread_id": "1", "recursion_limit": 5}}
    inputs = [
                HumanMessage(content=prompt)
                # HumanMessage(content='tell me about the Tu vu hospital in vietnam')
            ]
    state = {'messages': inputs, "chat_history": st.session_state.chat_history[-6:-1], "query": prompt}
    # result = app.invoke(state, config=config)
    answer = None
    for event in app.stream(state, config=config):
        print(event)
        for key, value in event.items():
            if value is None:
                continue
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint(value, indent=2, width=80, depth=None)
            print()
            if "messages" in value:
                answer = value["messages"][-1].content

    # Thêm phản hồi vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
