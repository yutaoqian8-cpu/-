from __future__ import annotations

from datetime import date

from storage import (
    add_application,
    application_metrics,
    delete_application,
    init_database,
    list_applications,
    update_application_status,
)


def test_application_lifecycle(tmp_path) -> None:
    database = tmp_path / "careerpilot.db"
    init_database(database)
    application_id = add_application(
        database,
        company="星海科技",
        role="数据分析实习生",
        applied_date=date(2026, 8, 2),
        status="已投递",
        score=82,
        skills=["SQL", "Python"],
        next_action="准备笔试",
    )
    frame = list_applications(database)
    assert len(frame) == 1
    assert frame.loc[0, "技能"] == "SQL、Python"
    assert application_metrics(frame)["active"] == 1

    update_application_status(database, application_id, "一面", "准备项目复盘")
    updated = list_applications(database)
    assert updated.loc[0, "进度"] == "一面"

    delete_application(database, application_id)
    assert list_applications(database).empty
