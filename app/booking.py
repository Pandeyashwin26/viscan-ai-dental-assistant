"""Deterministic synthetic scan-slot lookup and confirmation-gated booking."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import date, timedelta
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "appointments.json"


class BookingError(RuntimeError):
    """Base error for booking failures."""


class SlotUnavailable(BookingError):
    pass


class DuplicateBooking(BookingError):
    pass


class BookingStore:
    def __init__(self, appointments_path: Path | None = None):
        path = appointments_path or DATA_PATH
        self.slots = json.loads(path.read_text())
        self.bookings: dict[str, dict] = {}
        self._lock = threading.Lock()

    def available(self, scan_type: str | None = None, appointment_date: str | None = None) -> list[dict]:
        return [
            dict(slot)
            for slot in sorted(self.slots, key=lambda item: (item["date"], item["time"]))
            if slot["status"] == "available"
            and (not scan_type or slot["scan_type"] == scan_type)
            and (not appointment_date or slot["date"] == appointment_date)
        ]

    def book(self, slot_id: str, patient: dict) -> dict:
        with self._lock:
            slot = next((item for item in self.slots if item["id"] == slot_id), None)
            if not slot or slot["status"] != "available":
                raise SlotUnavailable("That appointment slot is no longer available.")
            if any(
                item["patient_id"] == patient["id"]
                and item["date"] == slot["date"]
                and item["scan_type"] == slot["scan_type"]
                for item in self.bookings.values()
            ):
                raise DuplicateBooking("This patient already has that scan type booked on that date.")
            booking_id = f"B-{uuid.uuid4().hex[:8].upper()}"
            booking = {
                "booking_id": booking_id,
                "slot_id": slot["id"],
                "patient_id": patient["id"],
                "patient_name": patient["name"],
                "date": slot["date"],
                "time": slot["time"],
                "scan_type": slot["scan_type"],
                "status": "confirmed",
            }
            slot["status"] = "booked"
            slot["patient_id"] = patient["id"]
            self.bookings[booking_id] = booking
            return dict(booking)


def resolve_relative_date(value: str | None, today: date | None = None) -> str | None:
    if not value:
        return None
    if value == "tomorrow":
        return ((today or date.today()) + timedelta(days=1)).isoformat()
    return value
