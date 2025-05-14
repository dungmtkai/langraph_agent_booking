import logging
import time

import streamlit as st
from datetime import datetime

from langchain_core.messages import HumanMessage

from agentsv2 import app
from utils import pretty_print_messages, parse_langgraph_output  # bạn có thể sửa lại để trả text thay vì print
from agents import supervisor  # thay bằng module thật sự của bạn

# Cấu hình giao diện
st.set_page_config(page_title="30Shine Chatbot", page_icon="💬")
st.title("💇‍♂️ 30Shine Chatbot Assistant")
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello anh, em là Janie trợ lý cá nhân của anh tại 30Shine. Em có thể giúp gì cho anh nè?"}
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
    #     with st.spinner("Processing..."):
    #         inputs = {
    #             "messages": [
    #                 HumanMessage(
    #                     content=prompt
    #                 )
    #             ],
    #         }
    #
    # last_msg = None

    config = {"configurable": {"thread_id": "1", "recursion_limit": 5}}
    inputs = [
                HumanMessage(content=prompt)
                # HumanMessage(content='tell me about the Tu vu hospital in vietnam')
            ]
    state = {'messages': inputs}
    result = app.invoke(state, config=config)


    # Thêm phản hồi vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": result["messages"][-1].content})
    with st.chat_message("assistant"):
        st.write(result["messages"][-1].content)
