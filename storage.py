from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

STATUSES = ["准备投递", "已投递", "笔试/测评", "一面", "二面/终面", "Offer", "暂缓", "已结束"]


def init_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                applied_date TEXT NOT NULL,
                status TEXT NOT NULL,
                score INTEGER,
                skills_json TEXT NOT NULL DEFAULT '[]',
                url TEXT NOT NULL DEFAULT '',
                next_action TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def add_application(
    path: Path,
    *,
    company: str,
    role: str,
    applied_date: date,
    status: str,
    score: int | None = None,
    skills: list[str] | None = None,
    url: str = "",
    next_action: str = "",
    notes: str = "",
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO applications (
                company, role, applied_date, status, score, skills_json,
                url, next_action, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company.strip(),
                role.strip(),
                applied_date.isoformat(),
                status if status in STATUSES else STATUSES[0],
                score,
                json.dumps(skills or [], ensure_ascii=False),
                url.strip(),
                next_action.strip(),
                notes.strip(),
                now,
                now,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_applications(path: Path) -> pd.DataFrame:
    with closing(sqlite3.connect(path)) as connection:
        frame = pd.read_sql_query(
            """
            SELECT id, company AS 公司, role AS 岗位, applied_date AS 日期,
                   status AS 进度, score AS 匹配信号, skills_json AS 技能,
                   url AS 链接, next_action AS 下一步, notes AS 备注
            FROM applications ORDER BY applied_date DESC, id DESC
            """,
            connection,
        )
    if not frame.empty:
        frame["技能"] = frame["技能"].map(lambda value: "、".join(json.loads(value or "[]")))
    return frame


def update_application_status(path: Path, application_id: int, status: str, next_action: str = "") -> None:
    if status not in STATUSES:
        raise ValueError("未知投递状态。")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE applications SET status = ?, next_action = ?, updated_at = ? WHERE id = ?",
            (status, next_action.strip(), datetime.now().isoformat(timespec="seconds"), application_id),
        )
        connection.commit()


def delete_application(path: Path, application_id: int) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM applications WHERE id = ?", (application_id,))
        connection.commit()


def application_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"total": 0, "active": 0, "interviews": 0, "offers": 0, "response_rate": 0}
    active_statuses = {"已投递", "笔试/测评", "一面", "二面/终面"}
    responded = {"笔试/测评", "一面", "二面/终面", "Offer", "已结束"}
    return {
        "total": len(frame),
        "active": int(frame["进度"].isin(active_statuses).sum()),
        "interviews": int(frame["进度"].isin({"一面", "二面/终面"}).sum()),
        "offers": int((frame["进度"] == "Offer").sum()),
        "response_rate": round(100 * frame["进度"].isin(responded).sum() / len(frame)),
    }
