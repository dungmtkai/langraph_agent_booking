# City IDs for salon lookup
CITY_IDS = {
    'Hà Giang': 0,
    'Thủ Đức': 1,
    'Hồ Chí Minh': 1,
    'Tiền Giang': 7,
    'Thanh Hóa': 9,
    'Thái Nguyên': 10,
    'Quảng Ninh': 16,
    'Nghệ An': 24,
    'Long An': 26,
    'Khánh Hòa': 33,
    'Hà Tĩnh': 38,
    'Đồng Nai': 42,
    'Bình Thuận': 48,
    'Bình Dương': 50,
    'Bình Định': 51,
    'Bắc Ninh': 53,
    'Bà Rịa - Vũng Tàu': 57,
    'An Giang': 58,
    'Hải Phòng': 59,
    'Đà Nẵng': 60,
    'Cần Thơ': 61,
    'Hà Nội': 62
}
BRANCH_HOURS = {
    "Cơ sở Hà Nội": {"start": "08:00", "end": "20:00", "interval": 60},
    "Cơ sở TP.HCM": {"start": "09:00", "end": "18:00", "interval": 30},
    "Cơ sở Đà Nẵng": {"start": "10:00", "end": "17:00", "interval": 60},
}

BOOKING_SYSTEM_PROMPT = (
    "Today is: {date_time}"
    "You are a booking agent.\n\n"
    "INSTRUCTIONS:\n"
    "- You can help users book a haircut, cancel existing appointments, check for available time slots, "
    "suggest salons near the user's location, or list available branches.\n"
    "- ALWAYS choose exactly ONE and ONLY ONE tool that best fits the user's request.\n"
    "- Do NOT use multiple tools in a single turn.\n"
    "- Use the tool that directly handles the request without assuming or over-processing.\n"
    "- After completing the task, respond only with the result — NO additional commentary.\n"
    "- If the task is unclear or unsupported, respond with an appropriate error using NO tool.\n"
)



