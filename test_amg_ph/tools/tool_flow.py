from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random


def run_cancel_leave_request(old_dates: List[str]) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Da huy nghi phep thanh cong",
        "cancelled_dates": old_dates,
        "cancelled_count": len(old_dates),
        "timestamp": datetime.now().isoformat()
    }


def run_cancel_medicine_ticket(old_dates: List[str]) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Da huy don dan thuoc thanh cong",
        "cancelled_dates": old_dates,
        "cancelled_count": len(old_dates),
        "timestamp": datetime.now().isoformat()
    }


def run_cancel_pickup_ticket(old_dates: List[str]) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Da huy don dan don thanh cong",
        "cancelled_dates": old_dates,
        "cancelled_count": len(old_dates),
        "timestamp": datetime.now().isoformat()
    }


def run_create_leave_ticket(
    reason: str,
    start_leave_datetime: str,
    number_of_days: int,
    dates: List[str]
) -> Dict[str, Any]:
    ticket_id = f"LV-{random.randint(1000, 9999)}"
    return {
        "success": True,
        "message": "Tao don xin nghi thanh cong",
        "ticket_id": ticket_id,
        "status": "created",
        "data": {
            "reason": reason,
            "start_leave_datetime": start_leave_datetime,
            "number_of_days": number_of_days,
            "dates": dates
        },
        "timestamp": datetime.now().isoformat()
    }


def run_create_medicine_ticket(
    dates: List[str],
    parent_note: str,
    medicine_images: Optional[List[str]] = None
) -> Dict[str, Any]:
    ticket_id = f"MD-{random.randint(1000, 9999)}"
    return {
        "success": True,
        "message": "Tao don dan thuoc thanh cong",
        "ticket_id": ticket_id,
        "status": "created",
        "data": {
            "dates": dates,
            "parent_note": parent_note,
            "medicine_images": medicine_images or []
        },
        "timestamp": datetime.now().isoformat()
    }


def run_create_pickup_ticket(
    dates: List[str],
    pickup_instructions: str,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    ticket_id = f"PU-{random.randint(1000, 9999)}"
    return {
        "success": True,
        "message": "Tao don dan don thanh cong",
        "ticket_id": ticket_id,
        "status": "created",
        "data": {
            "dates": dates,
            "pickup_instructions": pickup_instructions,
            "notes": notes or ""
        },
        "timestamp": datetime.now().isoformat()
    }


def run_edit_leave_ticket(
    old_dates: List[str],
    dates: List[str],
    reason: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Cap nhat don xin nghi thanh cong",
        "ticket_id": f"LV-{random.randint(1000, 9999)}",
        "status": "updated",
        "changes": {
            "old_dates": old_dates,
            "new_dates": dates,
            "reason": reason
        },
        "timestamp": datetime.now().isoformat()
    }


def run_edit_medicine_ticket(
    old_dates: List[str],
    dates: List[str],
    parent_note: Optional[str] = None,
    medicine_images: Optional[List[str]] = None
) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Cap nhat don dan thuoc thanh cong",
        "ticket_id": f"MD-{random.randint(1000, 9999)}",
        "status": "updated",
        "changes": {
            "old_dates": old_dates,
            "new_dates": dates,
            "parent_note": parent_note,
            "medicine_images": medicine_images
        },
        "timestamp": datetime.now().isoformat()
    }


def run_edit_pickup_ticket(
    old_dates: List[str],
    dates: List[str],
    pickup_instructions: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Cap nhat don dan don thanh cong",
        "ticket_id": f"PU-{random.randint(1000, 9999)}",
        "status": "updated",
        "changes": {
            "old_dates": old_dates,
            "new_dates": dates,
            "pickup_instructions": pickup_instructions
        },
        "timestamp": datetime.now().isoformat()
    }


def run_get_code_ticket() -> Dict[str, Any]:
    return {
        "success": True,
        "tickets": [
            {"ticket_id": "LV-1234", "type": "leave", "status": "approved", "date": "2024-12-10"},
            {"ticket_id": "MD-5678", "type": "medicine", "status": "pending", "date": "2024-12-11"},
            {"ticket_id": "PU-9012", "type": "pickup", "status": "approved", "date": "2024-12-12"}
        ],
        "total_count": 3,
        "timestamp": datetime.now().isoformat()
    }


def run_get_learning_schedule() -> Dict[str, Any]:
    return {
        "success": True,
        "schedule": [
            {"day": "Monday", "subjects": ["Toan", "Van", "Anh", "The duc"]},
            {"day": "Tuesday", "subjects": ["Toan", "Khoa hoc", "Lich su", "Am nhac"]},
            {"day": "Wednesday", "subjects": ["Van", "Anh", "Dia ly", "My thuat"]},
            {"day": "Thursday", "subjects": ["Toan", "Van", "Tin hoc", "The duc"]},
            {"day": "Friday", "subjects": ["Anh", "Khoa hoc", "Dao duc", "Sinh hoat"]}
        ],
        "semester": "HK1 2024-2025",
        "class_name": "5A1",
        "timestamp": datetime.now().isoformat()
    }


def run_get_menu() -> Dict[str, Any]:
    today = datetime.now()
    return {
        "success": True,
        "menu": [
            {
                "date": today.strftime("%Y-%m-%d"),
                "breakfast": "Phở bò, sữa tươi",
                "lunch": "Cơm, cá kho, canh rau, trái cây",
                "snack": "Bánh mì, sữa chua"
            },
            {
                "date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                "breakfast": "Bún riêu, nước ép cam",
                "lunch": "Cơm, gà rán, rau luộc, chè trôi nước",
                "snack": "Xôi, sữa đậu nành",
            }
        ],
        "week": today.isocalendar()[1],
        "timestamp": datetime.now().isoformat()
    }


def run_get_student_status() -> Dict[str, Any]:
    return {
        "success": True,
        "status": {
            "student_name": "Nguyen Van A",
            "student_id": "HS-2024-001",
            "class": "5A1",
            "attendance_today": "present",
            "health_status": "good",
            "recent_activities": [
                "Hoan thanh bai tap toan",
                "Tham gia hoat dong ngoai khoa"
            ],
            "teacher_notes": "Hoc tot, tich cuc trong lop"
        },
        "timestamp": datetime.now().isoformat()
    }


def run_get_ticket_summary_month() -> Dict[str, Any]:
    return {
        "success": True,
        "summary": {
            "month": datetime.now().month,
            "year": datetime.now().year,
            "leave_tickets": {"total": 3, "approved": 2, "pending": 1, "rejected": 0},
            "medicine_tickets": {"total": 5, "approved": 4, "pending": 1, "rejected": 0},
            "pickup_tickets": {"total": 8, "approved": 7, "pending": 1, "rejected": 0},
            "total_tickets": 16
        },
        "timestamp": datetime.now().isoformat()
    }


def run_get_ticket_summary_quarter() -> Dict[str, Any]:
    current_quarter = (datetime.now().month - 1) // 3 + 1
    return {
        "success": True,
        "summary": {
            "quarter": current_quarter,
            "year": datetime.now().year,
            "leave_tickets": {"total": 10, "approved": 8, "pending": 1, "rejected": 1},
            "medicine_tickets": {"total": 15, "approved": 13, "pending": 2, "rejected": 0},
            "pickup_tickets": {"total": 25, "approved": 23, "pending": 2, "rejected": 0},
            "total_tickets": 50
        },
        "timestamp": datetime.now().isoformat()
    }


def run_leave_date(dates: List[str]) -> Dict[str, Any]:
    today = datetime.now().date()
    valid_dates = []
    invalid_dates = []

    for date_str in dates:
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
            if date_obj >= today:
                valid_dates.append(date_str)
            else:
                invalid_dates.append({"date": date_str, "reason": "Ngay da qua"})
        except ValueError:
            invalid_dates.append({"date": date_str, "reason": "Dinh dang khong hop le"})

    return {
        "success": len(invalid_dates) == 0,
        "valid_dates": valid_dates,
        "invalid_dates": invalid_dates,
        "message": "Tat ca ngay hop le" if len(invalid_dates) == 0 else "Co ngay khong hop le",
        "timestamp": datetime.now().isoformat()
    }


def run_medicine_date(dates: List[str]) -> Dict[str, Any]:
    today = datetime.now().date()
    valid_dates = []
    invalid_dates = []

    for date_str in dates:
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
            if date_obj >= today:
                valid_dates.append(date_str)
            else:
                invalid_dates.append({"date": date_str, "reason": "Ngay da qua"})
        except ValueError:
            invalid_dates.append({"date": date_str, "reason": "Dinh dang khong hop le"})

    return {
        "success": len(invalid_dates) == 0,
        "valid_dates": valid_dates,
        "invalid_dates": invalid_dates,
        "message": "Tat ca ngay hop le" if len(invalid_dates) == 0 else "Co ngay khong hop le",
        "timestamp": datetime.now().isoformat()
    }


def run_pickup_date(dates: List[str]) -> Dict[str, Any]:
    today = datetime.now().date()
    valid_dates = []
    invalid_dates = []

    for date_str in dates:
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
            if date_obj >= today:
                valid_dates.append(date_str)
            else:
                invalid_dates.append({"date": date_str, "reason": "Ngay da qua"})
        except ValueError:
            invalid_dates.append({"date": date_str, "reason": "Dinh dang khong hop le"})

    return {
        "success": len(invalid_dates) == 0,
        "valid_dates": valid_dates,
        "invalid_dates": invalid_dates,
        "message": "Tat ca ngay hop le" if len(invalid_dates) == 0 else "Co ngay khong hop le",
        "timestamp": datetime.now().isoformat()
    }


TOOL_FUNCTIONS = {
    133: run_cancel_leave_request,
    134: run_cancel_medicine_ticket,
    135: run_cancel_pickup_ticket,
    136: run_create_leave_ticket,
    137: run_create_medicine_ticket,
    138: run_create_pickup_ticket,
    139: run_edit_leave_ticket,
    140: run_edit_medicine_ticket,
    141: run_edit_pickup_ticket,
    142: run_get_code_ticket,
    143: run_get_learning_schedule,
    144: run_get_menu,
    146: run_get_student_status,
    147: run_get_ticket_summary_month,
    148: run_get_ticket_summary_quarter,
    149: run_leave_date,
    150: run_medicine_date,
    151: run_pickup_date,
}
