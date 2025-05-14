import streamlit as st
from datetime import datetime
from utils import pretty_print_messages  # bạn có thể sửa lại để trả text thay vì print

from agents import supervisor  # thay 'your_module' bằng tên file gốc bạn đã viết code supervisor ở đó

# Streamlit app UI
st.set_page_config(page_title="30Shine Chatbot", page_icon="💬")
st.title("💇‍♂️ 30Shine Chatbot Assistant")

# Lưu lịch sử đoạn hội thoại
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Giao diện nhập tin nhắn
user_input = st.chat_input("Nhập câu hỏi hoặc yêu cầu của bạn...")

# Nếu người dùng gửi tin nhắn
if user_input:
    # Thêm câu người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(user_input)

    # Tạo input messages cho supervisor
    messages_input = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history]

    # Chạy supervisor
    full_response = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for id_tuple, chunk in supervisor.stream({"messages": messages_input}, subgraphs=True):
            messages = chunk.get('agent', {}).get('messages', [])
            if messages:
                last_msg = messages[-1].content
            # last_msg = chunk["supervisor"]["messages"]
            full_response.append(last_msg)
        message_placeholder.markdown(last_msg)

    print(full_response)

    # Thêm phản hồi của hệ thống vào lịch sử
    st.session_state.chat_history.append({"role": "assistant", "content": last_msg})
