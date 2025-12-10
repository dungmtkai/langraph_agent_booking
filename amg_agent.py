import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from test_amg_ph.agents import build_workflow, AGENT_NODE_TO_ID

load_dotenv()


# ========================= STREAMLIT UI =========================
st.set_page_config(page_title="AMG PH Agent", layout="wide")
st.title("AMG PH Agent")

# Sidebar for user info config
with st.sidebar:
    st.header("Thông tin phụ huynh")
    parent_name = st.text_input("Tên phụ huynh", value="Mai Thị Kim Dung")
    relationship = st.selectbox("Mối quan hệ", ["MOTHER", "FATHER", "GUARDIAN"])
    child_name = st.text_input("Tên học sinh", value="Nguyễn Thị Mai")
    nickname = st.text_input("Tên gọi ở nhà", value="Chíp")
    class_name = st.text_input("Lớp", value="Chồi BU1")
    teacher_name = st.text_input("Giáo viên chủ nhiệm", value="Nguyễn Thị Hà")

    st.divider()
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Build user_info from sidebar
user_info = {
    "parentName": parent_name,
    "relationship": relationship,
    "childName": child_name,
    "nickname": nickname,
    "className": class_name,
    "teacherName": teacher_name,
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "amg_ph_chat_001"
if "app" not in st.session_state:
    st.session_state.app = None

# Build app with user_info (rebuild if user_info changes)
@st.cache_resource
def get_app(user_info_str: str):
    import json
    user_info = json.loads(user_info_str)
    return build_workflow(user_info=user_info)

import json
app = get_app(json.dumps(user_info))

# Get all agent node names for checking response source
agent_node_names = list(AGENT_NODE_TO_ID.keys())

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input
if prompt := st.chat_input("Dạ anh/chị ơi, em nghe ạ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Đang xử lý..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt,
            "original_query": prompt,
            "chat_history": st.session_state.chat_history,
            "list_tasks": []
        }

        final_answer = None
        try:
            for chunk in app.stream(inputs, config=config, stream_mode="values"):
                msgs = chunk.get("messages", [])
                if msgs and msgs[-1].content:
                    last_msg = msgs[-1]
                    # Check if response is from an agent node
                    if hasattr(last_msg, "name") and last_msg.name in agent_node_names:
                        final_answer = last_msg.content
                    elif len(msgs) == 1 and not hasattr(last_msg, "name"):
                        final_answer = last_msg.content

            if final_answer:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))
                st.chat_message("assistant").write(final_answer)
            else:
                fallback = "Dạ anh/chị ơi, em chưa hiểu lắm, anh/chị nói rõ hơn được không ạ?"
                st.session_state.messages.append({"role": "assistant", "content": fallback})
                st.chat_message("assistant").write(fallback)

        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
            st.write("Chi tiết lỗi:", e)
