from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import json
import requests
from config import CITY_IDS
from utils import euclidean_distance
from openai import OpenAI

# Khởi tạo client OpenAI (thay bằng API key của bạn)
# openai_client = OpenAI(api_key="")

# Giả lập danh sách lịch hẹn
appointments = []


# Định nghĩa trạng thái của đồ thị
class GraphState(TypedDict):
    messages: List[Dict[str, Any]]  # Lưu trữ tin nhắn
    intent: str  # Ý định: booking hoặc info
    tasks: List[Dict[str, Any]]  # Danh sách tác vụ con
    current_task_index: int  # Chỉ số tác vụ hiện tại
    agent: str  # Agent hiện tại: booking hoặc infor
    tool_results: List[str]  # Kết quả từ các công cụ


# Prompt để gọi LLM phân tích truy vấn
LLM_ANALYSIS_PROMPT = """
Bạn là một trợ lý AI thông minh, hỗ trợ phân tích truy vấn từ người dùng để xác định ý định và trích xuất các tác vụ cần thực hiện. Hệ thống có hai agent:
- **booking_agent**: Xử lý các tác vụ như đặt lịch, hủy lịch, kiểm tra khung giờ trống, tìm salon gần nhất, liệt kê chi nhánh. Các công cụ:
  - list_branches: Liệt kê chi nhánh salon.
  - get_near_salon: Tìm salon gần nhất (yêu cầu user_address, city).
  - check_availability: Kiểm tra khung giờ trống (yêu cầu salon_address, date, time).
  - book_appointment: Đặt lịch hẹn (yêu cầu time, salon_address, date, booking_phone).
  - cancel_appointment: Hủy lịch (yêu cầu cancel_phone).
- **infor_agent**: Cung cấp thông tin về giá cả, dịch vụ, chi nhánh. Các công cụ:
  - faq_answer: Trả lời câu hỏi thường gặp (giá cả, dịch vụ).
  - get_near_salon: Tìm salon gần nhất.
  - list_branches: Liệt kê chi nhánh.

Dựa trên truy vấn của người dùng, hãy:
1. Xác định ý định: "booking" (liên quan đến đặt/hủy lịch, khung giờ, salon) hoặc "info" (liên quan đến giá cả, dịch vụ, thông tin salon).
2. Trích xuất danh sách tác vụ, mỗi tác vụ gồm:
   - tool: Tên công cụ cần gọi.
   - params: Tham số cho công cụ (giữ null nếu không rõ, ví dụ salon_address có thể được lấy từ get_near_salon sau).
3. Nếu thiếu thông tin (như ngày, giờ), sử dụng giá trị mặc định:
   - date: "15-05-2025"
   - time: "10:00"
   - booking_phone: "0123456789"
   - city: "Hà Nội" (nếu không rõ thành phố)
4. Đảm bảo các tác vụ được sắp xếp hợp lý (ví dụ, get_near_salon trước book_appointment nếu cần salon_address).

Truy vấn: {query}

Trả về JSON với định dạng:
```json
{
  "intent": "booking" hoặc "info",
  "tasks": [
    {
      "tool": "tên_công_cụ",
      "params": {
        "param_name": "giá_trị" hoặc null
      }
    },
    ...
  ]
}
```
"""


# Hàm gọi LLM để phân tích truy vấn
def analyze_query_with_llm(query: str) -> Dict[str, Any]:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LLM_ANALYSIS_PROMPT.format(query=query)},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # Fallback nếu LLM thất bại
        return {
            "intent": "info",
            "tasks": [{"tool": "faq_answer", "params": {}}]
        }


def collect_booking_info(
        user_address: str | None = None,
        date: str | None = None,
        time: str | None = None,
        phone: str | None = None
) -> list[str]:
    """Collect missing booking information from the user."""
    messages = []
    if not user_address:
        messages.append(
            "Dạ, hệ thống bên em có hơn 100 chi nhánh trên khắp cả nước, như Hà Nội, Hồ Chí Minh, Hải Phòng, Bình Dương, Vinh, Đồng Nai... Anh ở khu vực nào để em giúp tìm salon gần nhất"
        )
    if not date:
        messages.append("Bạn muốn đặt lịch vào ngày nào? (Định dạng: DD-MM-YYYY)")
    if not time:
        messages.append("Bạn muốn đặt lịch vào khung giờ nào? (Định dạng: HH:MM, từ 08:00 đến 20:00)")
    if not phone:
        messages.append("Vui lòng cung cấp số điện thoại của bạn để xác nhận lịch hẹn.")
    return messages


@tool
def list_branches() -> str:
    """List available salon branches."""
    url = "https://storage.30shine.com/web/v3/configs/get_all_salon.json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = json.loads(response.content.decode('utf-8-sig'))
        return (
            f"Hiện tại bên em đang có {int(data['count'])} chi nhánh khác nhau trên khắp cả nước như "
            "Hà Nội, Hồ Chí Minh, Hải Phòng, Bình Dương, Vinh, Đồng Nai. "
            "Anh ở khu vực nào để em giúp tìm salon gần nhất?"
        )
    except (requests.RequestException, json.JSONDecodeError):
        return "Dạ xin lỗi, em không thể cung cấp thông tin này."


@tool(parse_docstring=True)
def get_near_salon(user_address: str, city: str) -> str:
    """
    Suggest the nearest salon based on user address and city.

    Args:
        user_address (str): The street address or specific location provided by the user.
        city (str): The city name where the user is located.

    Returns:
        str: Result of the tool.
    """
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={user_address}+{city}&apiKey=A7V_JCsxV2Y_A_WBg00q_mUB-bDCynwEhwaZeT6QfwY&limit=1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = json.loads(response.content.decode('utf-8-sig'))
        res = data["items"][0]
        city_id = CITY_IDS.get(res['address']['county'])
        if city_id is None:
            return "Không tìm thấy thành phố phù hợp. Vui lòng thử lại."

        lat_lon = res['position']
        near_salon = {'city_id': city_id, 'lat': lat_lon['lat'], 'lon': lat_lon['lng']}

        url_get_all_salon = "https://storage.30shine.com/web/v3/configs/get_all_salon.json"
        response = requests.get(url_get_all_salon, timeout=5)
        response.raise_for_status()
        data = json.loads(response.content.decode('utf-8-sig'))

        salons = [x for x in data["data"] if x["cityId"] == near_salon['city_id']]
        salons.sort(
            key=lambda x: euclidean_distance(
                near_salon['lat'], near_salon['lon'], x['latitude'], x['longitude']
            )
        )

        if not salons:
            return "Không tìm thấy salon nào gần khu vực của bạn."

        list_salon = "Danh sách salon\n" + "\n".join(
            f"- **{x['addressNew']}**" for x in salons[:5]
        )
        return list_salon
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return "Dạ xin lỗi, em không thể cung cấp thông tin này."


@tool(parse_docstring=True)
def check_availability(salon_address: str, date: str, time: str):
    """
    Check available time slots for a specific branch and date.

    Args:
        salon_address (str): The address of the salon branch.
        date (str): The date of the appointment in 'DD-MM-YYYY' format.
        time (str): The time of the appointment in 'HH:MM' format. Must be between 08:00 and 20:00.

    Returns:
        str: Result of the tool.
    """

    # Get salon id
    url = f"https://storage.30shine.com/web/v3/configs/get_all_salon.json?"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = json.loads(response.content.decode('utf-8-sig'))
        all_salon = data["data"]
        id_salon = None
        for salon in all_salon:
            if salon["addressNew"] == salon_address:
                id_salon = salon["id"]
                break

        # Check available slot
        check_slot_url = f"https://3sgus10dig.execute-api.ap-southeast-1.amazonaws.com/Prod/booking-view-service/api/v1/booking/book-hours-group?salonId={id_salon}&bookDate={date}&timeRequest={time}"

        try:
            response_slot = requests.get(check_slot_url, timeout=5)
            response_slot.raise_for_status()
            data_slot = json.loads(response_slot.content.decode('utf-8-sig'))
            list_hours = data_slot["data"]["hourGroup"]

            # Tách phần giờ và phút từ chuỗi thời gian
            hour, minute = time.split(':')  # Lấy giờ và phút
            # Định dạng lại ngày giờ
            hour = hour.lstrip("0")
            if int(minute) % 20 != 0:
                convert_minute = (int(minute) // 20) * 20
                if convert_minute == 0:
                    minute = str(convert_minute) + "0"
                else:
                    minute = str((int(minute) // 20) * 20)

            hour_minute = f"{hour}h{minute}"  # Định dạng giờ phút theo chuẩn

            # Tìm nhóm giờ hiện tại
            find_hour_group = next(
                (hourGroup for hourGroup in list_hours if hourGroup['name'] == hour),
                None
            )

            # Kết quả trả về
            return_response = {"isFree": False, "hourId": "", "subHourId": "", "nearest_free_before": None,
                               "nearest_free_after": None}

            if find_hour_group:
                # Tìm thời gian khớp trong nhóm giờ hiện tại
                time_matching = next(
                    (hour for hour in find_hour_group['hours'] if hour['hour'] == hour_minute),
                    None
                )

                # Kiểm tra nếu `time_matching` tồn tại và có `isFree = True`
                if time_matching:
                    return_response["isFree"] = time_matching["isFree"]
                    return_response["hourId"] = time_matching["hourId"]
                    return_response["subHourId"] = time_matching["subHourId"]

                    # Lấy giờ hiện tại
                    current_hour = int(hour)

                    def time_to_minutes(time_str):
                        """
                        Chuyển chuỗi giờ dạng 'hhhmm' thành số phút kể từ 00:00.
                        """
                        hour, minute = map(int, time_str.replace('h', ':').split(':'))
                        return hour * 60 + minute

                    # Hàm tìm giờ gần nhất trống trong danh sách
                    def find_nearest_free(hours, target_hour):
                        target_minutes = time_to_minutes(target_hour)
                        before = after = None
                        for h in hours:
                            if h["isFree"]:
                                current_minutes = time_to_minutes(h["hour"])
                                if current_minutes < target_minutes:
                                    before = h
                                elif current_minutes > target_minutes and after is None:
                                    after = h
                        return before, after

                    # Tìm giờ trống trong phạm vi 1 tiếng liền kề
                    relevant_hour_groups = [
                        group for group in list_hours
                        if current_hour - 4 <= int(group["name"]) <= current_hour + 4
                    ]
                    all_hours = [hour for group in relevant_hour_groups for hour in group["hours"]]
                    nearest_before, nearest_after = find_nearest_free(all_hours, time_matching["hour"])

                    # Gán kết quả giờ gần nhất trước và sau vào response
                    return_response["nearest_free_before_booked_time"] = {
                        "hourFrame": nearest_before["hourFrame"],
                        "hourId": nearest_before["hourId"],
                        "subHourId": nearest_before["subHourId"],
                    } if nearest_before else None

                    return_response["nearest_free_after_booked_time"] = {
                        "hourFrame": nearest_after["hourFrame"],
                        "hourId": nearest_after["hourId"],
                        "subHourId": nearest_after["subHourId"],
                    } if nearest_after else None
            if return_response["isFree"]:
                return "Còn slot"

            else:
                before_time = return_response["nearest_free_before_booked_time"]["hourFrame"]
                after_time = return_response["nearest_free_after_booked_time"]["hourFrame"]
                return f"Hết slot. Hai khung giờ gần nhất còn slot là {before_time} và {after_time}"

        except (requests.RequestException, json.JSONDecodeError, KeyError):
            return "Dạ xin lỗi, em không thể cung cấp thông tin này."
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return "Dạ xin lỗi, em không thể cung cấp thông tin này."


@tool(parse_docstring=True)
def book_appointment(
        time: str,
        salon_address: str,
        date: str,
        booking_phone: str
) -> str:
    """
    Book a haircut appointment.

    Args:
        time (str): The time of the appointment in 'HH:MM'. Must be between 08:00 and 20:00.
        salon_address (str): The address of the salon branch.
        date (str): The date of the appointment in 'DD-MM-YYYY' format.
        booking_phone (str): The user's phone number for confirming the appointment.

    Returns:
        str: Result of the tool.
    """

    if not salon_address:
        return "Dạ, hệ thống bên em có hơn 100 chi nhánh trên khắp cả nước, như Hà Nội, Hồ Chí Minh, Hải Phòng, Bình Dương, Vinh, Đồng Nai... Anh ở khu vực nào để em giúp tìm salon gần nhất"
    # if not all([branch, date, time, phone]):
    #     result = collect_booking_info(branch, date, time, phone)
    #     texts = [msg.content if isinstance(msg.content, str) else msg.content.text
    #              for msg in result.messages]
    #     return "\n".join(texts)

    try:
        hour = int(time.split(":")[0])
        if hour < 8 or hour > 20:
            return "Giờ đặt không hợp lệ. Vui lòng chọn khung giờ từ 08:00 đến 20:00."
    except ValueError:
        return "Định dạng giờ không hợp lệ. Vui lòng sử dụng định dạng HH:MM."

    for appt in appointments:
        if appt["branch"] == salon_address and appt["date"] == date and appt["time"] == time:
            return "Khung giờ này đã được đặt. Vui lòng chọn khung giờ khác."

    appointments.append({
        "branch": salon_address,
        "date": date,
        "time": time,
        "phone": booking_phone
    })
    return f"Đã đặt lịch thành công tại {salon_address} vào {date} lúc {time} cho số điện thoại {booking_phone}."

@tool(parse_docstring=True)
def cancel_appointment(cancel_phone: str) -> str:
    """
        Cancel an appointment based on phone number.

        Args:
            cancel_phone (str): The number phone that user used to cancel apointment

        Returns:
            str: Result of the tool.
        """
    if cancel_phone == "0366761395":
        return f"Đã hủy lịch hẹn cho số điện thoại {cancel_phone}."
    return f"Không tìm thấy lịch hẹn cho số điện thoại {cancel_phone}."


@tool()
def faq_answer():
    """Answer question about 30Shine"""
    return "Giá cắt tóc ở 30 Shine là 100 nghìn"



# Hàm của supervisor agent
def supervisor_node(state: GraphState) -> GraphState:
    query = state["messages"][-1]["content"]
    analysis = analyze_query_with_llm(query)

    state["intent"] = analysis["intent"]
    state["tasks"] = analysis["tasks"]
    state["current_task_index"] = 0
    state["agent"] = "booking" if analysis["intent"] == "booking" else "infor"
    state["tool_results"] = []

    return state


# Hàm của booking agent
def booking_agent_node(state: GraphState) -> GraphState:
    current_task_index = state["current_task_index"]
    tasks = state["tasks"]

    if current_task_index >= len(tasks):
        return state  # Không còn tác vụ

    task = tasks[current_task_index]
    tool_name = task["tool"]
    params = task["params"]

    # Xử lý tham số thiếu (ví dụ, salon_address)
    if tool_name in ["check_availability", "book_appointment"] and not params.get("salon_address"):
        for msg in reversed(state["messages"]):
            if msg["role"] == "tool" and isinstance(msg["content"], str) and msg["content"].startswith(
                    "Danh sách salon"):
                first_salon = msg["content"].split("\n")[1].strip("- **")
                params["salon_address"] = first_salon
                break

    # Gọi công cụ
    try:
        if tool_name == "list_branches":
            result = list_branches()
        elif tool_name == "get_near_salon":
            result = get_near_salon(**params)
            if result.startswith("Danh sách salon"):
                first_salon = result.split("\n")[1].strip("- **")
                state["messages"].append({"role": "tool", "content": {"salon_address": first_salon}})
        elif tool_name == "check_availability":
            result = check_availability(**params)
        elif tool_name == "book_appointment":
            result = book_appointment(**params)
        elif tool_name == "cancel_appointment":
            result = cancel_appointment(**params)
        else:
            result = "Công cụ không hợp lệ."
    except TypeError:
        result = "Thiếu hoặc sai tham số cho công cụ."

    state["tool_results"].append(result)
    state["current_task_index"] += 1
    state["messages"].append({"role": "tool", "content": result})

    return state


# Hàm của infor agent
def infor_agent_node(state: GraphState) -> GraphState:
    current_task_index = state["current_task_index"]
    tasks = state["tasks"]

    if current_task_index >= len(tasks):
        return state  # Không còn tác vụ

    task = tasks[current_task_index]
    tool_name = task["tool"]
    params = task["params"]

    # Gọi công cụ
    try:
        if tool_name == "faq_answer":
            result = faq_answer()
        elif tool_name == "get_near_salon":
            result = get_near_salon(**params)
        elif tool_name == "list_branches":
            result = list_branches()
        else:
            result = "Công cụ không hợp lệ."
    except TypeError:
        result = "Thiếu hoặc sai tham số cho công cụ."

    state["tool_results"].append(result)
    state["current_task_index"] += 1
    state["messages"].append({"role": "tool", "content": result})

    return state


# Hàm định tuyến
def router(state: GraphState) -> str:
    if state["current_task_index"] < len(state["tasks"]):
        return state["agent"] + "_agent"
    return "finalize"


# Hàm hoàn thiện kết quả
def finalize_node(state: GraphState) -> GraphState:
    results = state["tool_results"]
    final_response = "\n".join(results) if results else "Không có kết quả."
    state["messages"].append({"role": "assistant", "content": final_response})
    return state


# Xây dựng đồ thị
workflow = StateGraph(GraphState)

# Thêm các nút
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("booking_agent", booking_agent_node)
workflow.add_node("infor_agent", infor_agent_node)
workflow.add_node("finalize", finalize_node)

# Thêm các cạnh
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["agent"] + "_agent",
    {
        "booking_agent": "booking_agent",
        "infor_agent": "infor_agent"
    }
)
workflow.add_conditional_edges(
    "booking_agent",
    router,
    {
        "booking_agent": "booking_agent",
        "infor_agent": "infor_agent",
        "finalize": "finalize"
    }
)
workflow.add_conditional_edges(
    "infor_agent",
    router,
    {
        "booking_agent": "booking_agent",
        "infor_agent": "infor_agent",
        "finalize": "finalize"
    }
)
workflow.add_edge("finalize", END)

# Đặt điểm bắt đầu
workflow.set_entry_point("supervisor")

# Biên dịch đồ thị
graph = workflow.compile()


# Hàm chạy đồ thị
def run_graph(query: str) -> str:
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "intent": "",
        "tasks": [],
        "current_task_index": 0,
        "agent": "",
        "tool_results": []
    }
    result = graph.invoke(initial_state)
    return result["messages"][-1]["content"]


# Ví dụ sử dụng
if __name__ == "__main__":
    query = "Tìm salon gần nhất tại Hà Nội và đặt lịch vào ngày 15-05-2025 lúc 10:00 số điện thoại 0123456789"
    response = run_graph(query)
    print(response)