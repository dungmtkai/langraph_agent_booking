from datetime import datetime, timedelta
from random import random
from typing import Optional

from langchain_core.tools import tool
import json
import requests
from config import CITY_IDS
from utils import euclidean_distance

appointments = []


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
    Check available time slots for a specific branch and date. All these information must be acquired from user.

    Args:
        salon_address (str): The address of the salon branch.
        date (str): The date of the appointment in 'DD-MM-YYYY' format.
        time (str): The time of the appointment in 'HH:MM' format. Must be between 08:00 and 20:00.

    Returns:
        str: Result of the tool.
    """
    import random
    random_slot = random.random()
    fmt = "%H:%M"
    time_obj = datetime.strptime(time, fmt)

    if time == "10:00":
        # Lấy thời gian trước và sau 20 phút
        time_before = time_obj - timedelta(minutes=20)
        time_after = time_obj + timedelta(minutes=20)
        return f"Ở {salon_address} ngày {date} lúc {time} đã hết slot. Hai khung giờ gần nhất còn slot là {time_before.strftime(fmt)} và {time_after.strftime(fmt)}"
    else:
        return f"Ở {salon_address} ngày {date} lúc {time} còn slot"
    # # Get salon id
    # url = f"https://storage.30shine.com/web/v3/configs/get_all_salon.json?"
    # try:
    #     response = requests.get(url, timeout=5)
    #     response.raise_for_status()
    #     data = json.loads(response.content.decode('utf-8-sig'))
    #     all_salon = data["data"]
    #     id_salon = None
    #     for salon in all_salon:
    #         if salon_address.lower() in salon["addressNew"].lower():
    #             id_salon = salon["id"]
    #             break
    #
    #     # Check available slot
    #     check_slot_url = f"https://3sgus10dig.execute-api.ap-southeast-1.amazonaws.com/Prod/booking-view-service/api/v1/booking/book-hours-group?salonId={id_salon}&bookDate={date}&timeRequest={time}"
    #
    #     try:
    #         response_slot = requests.get(check_slot_url, timeout=5)
    #         response_slot.raise_for_status()
    #         data_slot = json.loads(response_slot.content.decode('utf-8-sig'))
    #         list_hours = data_slot["data"]["hourGroup"]
    #
    #         # Tách phần giờ và phút từ chuỗi thời gian
    #         hour, minute = time.split(':')  # Lấy giờ và phút
    #         # Định dạng lại ngày giờ
    #         hour = hour.lstrip("0")
    #         if int(minute) % 20 != 0:
    #             convert_minute = (int(minute) // 20) * 20
    #             if convert_minute == 0:
    #                 minute = str(convert_minute) + "0"
    #             else:
    #                 minute = str((int(minute) // 20) * 20)
    #
    #         hour_minute = f"{hour}h{minute}"  # Định dạng giờ phút theo chuẩn
    #
    #         # Tìm nhóm giờ hiện tại
    #         find_hour_group = next(
    #             (hourGroup for hourGroup in list_hours if hourGroup['name'] == hour),
    #             None
    #         )
    #
    #         # Kết quả trả về
    #         return_response = {"isFree": False, "hourId": "", "subHourId": "", "nearest_free_before": None,
    #                            "nearest_free_after": None}
    #
    #         if find_hour_group:
    #             # Tìm thời gian khớp trong nhóm giờ hiện tại
    #             time_matching = next(
    #                 (hour for hour in find_hour_group['hours'] if hour['hour'] == hour_minute),
    #                 None
    #             )
    #
    #             # Kiểm tra nếu `time_matching` tồn tại và có `isFree = True`
    #             if time_matching:
    #                 return_response["isFree"] = time_matching["isFree"]
    #                 return_response["hourId"] = time_matching["hourId"]
    #                 return_response["subHourId"] = time_matching["subHourId"]
    #
    #                 # Lấy giờ hiện tại
    #                 current_hour = int(hour)
    #
    #                 def time_to_minutes(time_str):
    #                     """
    #                     Chuyển chuỗi giờ dạng 'hhhmm' thành số phút kể từ 00:00.
    #                     """
    #                     hour, minute = map(int, time_str.replace('h', ':').split(':'))
    #                     return hour * 60 + minute
    #
    #                 # Hàm tìm giờ gần nhất trống trong danh sách
    #                 def find_nearest_free(hours, target_hour):
    #                     target_minutes = time_to_minutes(target_hour)
    #                     before = after = None
    #                     for h in hours:
    #                         if h["isFree"]:
    #                             current_minutes = time_to_minutes(h["hour"])
    #                             if current_minutes < target_minutes:
    #                                 before = h
    #                             elif current_minutes > target_minutes and after is None:
    #                                 after = h
    #                     return before, after
    #
    #                 # Tìm giờ trống trong phạm vi 1 tiếng liền kề
    #                 relevant_hour_groups = [
    #                     group for group in list_hours
    #                     if current_hour - 4 <= int(group["name"]) <= current_hour + 4
    #                 ]
    #                 all_hours = [hour for group in relevant_hour_groups for hour in group["hours"]]
    #                 nearest_before, nearest_after = find_nearest_free(all_hours, time_matching["hour"])
    #
    #                 # Gán kết quả giờ gần nhất trước và sau vào response
    #                 return_response["nearest_free_before_booked_time"] = {
    #                     "hourFrame": nearest_before["hourFrame"],
    #                     "hourId": nearest_before["hourId"],
    #                     "subHourId": nearest_before["subHourId"],
    #                 } if nearest_before else None
    #
    #                 return_response["nearest_free_after_booked_time"] = {
    #                     "hourFrame": nearest_after["hourFrame"],
    #                     "hourId": nearest_after["hourId"],
    #                     "subHourId": nearest_after["subHourId"],
    #                 } if nearest_after else None
    #         if return_response["isFree"]:
    #             return "Còn slot"
    #
    #         else:
    #             before_time = return_response["nearest_free_before_booked_time"]["hourFrame"]
    #             after_time = return_response["nearest_free_after_booked_time"]["hourFrame"]
    #             return f"Hết slot. Hai khung giờ gần nhất còn slot là {before_time} và {after_time}"
    #
    #     except (requests.RequestException, json.JSONDecodeError, KeyError):
    #         return "Còn slot"
    # except (requests.RequestException, json.JSONDecodeError, KeyError):
    #     return "Còn slot"


from pydantic import BaseModel, Field


@tool()
def confirm_booking():
    """Confirm information booking before booking"""
    return "Anh chị xác nhận thông tin đặt lịch với các thông tin đã cung cấp giúp em ạ"

class BookingInput(BaseModel):
    time: str = Field(..., description="The time of the appointment in 'HH:MM'. Must be between 08:00 and 20:00.")
    salon_address: str = Field(..., description="The address of the salon branch.")
    date: str = Field(..., description="The date of the appointment in 'DD-MM-YYYY' format.")
    booking_phone: str = Field(..., description="The user's phone number for confirming the appointment.")
    otp: Optional[str] = Field(None, description="The OTP code for booking.")


@tool(args_schema=BookingInput)
def book_appointment(
        time: str,
        salon_address: str,
        date: str,
        booking_phone: str,
        otp: Optional[str] = None
) -> str:
    """
    Book a haircut appointment.
    """

    if not salon_address:
        return "Dạ, hệ thống bên em có hơn 100 chi nhánh trên khắp cả nước, như Hà Nội, Hồ Chí Minh, Hải Phòng, Bình Dương, Vinh, Đồng Nai... Anh ở khu vực nào để em giúp tìm salon gần nhất"
    if not booking_phone:
        return "Cho xin số điện thoại"
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

    # slot_check = check_availability(salon_address=salon_address, date=date, time=time)
    # if not "Còn slot":
    #     return f"Ôi tiếc quá {salon_address} vào {date} lúc {time} hiện đang hết slot. {slot_check}"
    # else:
    # Send mã OTP:

    if otp:
        return f"Đã đặt lịch thành công tại {salon_address} vào {date} lúc {time} cho số điện thoại {booking_phone} với mã OTP là : {otp}."
    return f"Em đã gữi mã OTP về số điện thoại {booking_phone} của anh chị. Anh chị vui lòng nhập mã OTP để em tiếp tục đặt lịch nhé."


@tool(parse_docstring=True)
def cancel_appointment(cancel_phone: str) -> str:
    """
        Cancel an appointment based on provide phone number.

        Args:
            cancel_phone (str): The number phone that user used to cancel apointment

        Returns:
            str: Result of the tool.
        """
    if cancel_phone == "0366761395":
        return f"Đã hủy lịch hẹn cho số điện thoại {cancel_phone}."
    return f"Không tìm thấy lịch hẹn cho số điện thoại {cancel_phone}."


@tool()
def get_info():
    """Provide about service, price of 30Shine"""
    return "Giá cắt tóc ở 30 Shine là 100 nghìn"
