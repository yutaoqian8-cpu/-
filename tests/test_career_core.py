from __future__ import annotations

import io

import pytest
from docx import Document

from career_core import (
    MAX_RESUME_BYTES,
    JobRequirements,
    analyze_match,
    extract_job_requirements,
    extract_resume_text,
    infer_resume_profile,
    score_interview_answer,
    top_skill_priorities,
)

RESUME = """某大学，本科，信息管理专业。
技能：Python、SQL、Excel、Pandas、数据分析。
项目：使用 Python 和 SQL 清洗 50000 条数据，制作经营分析看板，将周报时间缩短 60%。
"""

JOB = """岗位：数据分析实习生
公司：星海科技
岗位职责：负责业务数据分析，使用 SQL 提取数据并制作数据可视化看板。
任职要求：本科及以上学历，熟练掌握 SQL、Excel 和数据分析；熟悉 Python、Pandas 优先。
"""


def test_extract_job_requirements_separates_must_and_preferred() -> None:
    job = extract_job_requirements(JOB)
    assert job.title == "数据分析实习生"
    assert job.company == "星海科技"
    assert {"SQL", "Excel", "数据分析"}.issubset(job.must_skills)
    assert {"Python", "Pandas"}.issubset(job.preferred_skills)
    assert job.education == "本科"
    assert all(not item.startswith("岗位：") for item in job.responsibilities)


def test_match_score_uses_evidence_and_not_probability() -> None:
    profile = infer_resume_profile(RESUME, education="本科")
    analysis = analyze_match(profile, extract_job_requirements(JOB))
    assert analysis.score >= 70
    assert "SQL" in analysis.matched_skills
    assert 0 <= analysis.keyword_score <= 100


def test_missing_core_skill_is_reported() -> None:
    profile = infer_resume_profile("本科，擅长内容创作和新媒体运营。", education="本科")
    analysis = analyze_match(profile, extract_job_requirements(JOB))
    assert "SQL" in analysis.missing_skills
    assert analysis.score < 80


def test_education_hard_gap_caps_score() -> None:
    profile = infer_resume_profile(RESUME, education="大专")
    analysis = analyze_match(profile, extract_job_requirements(JOB))
    assert analysis.education_score == 0
    assert analysis.score <= 59


def test_short_job_description_is_rejected() -> None:
    with pytest.raises(ValueError, match="过短"):
        extract_job_requirements("招聘实习生")


def test_resume_size_limit() -> None:
    with pytest.raises(ValueError, match="8 MB"):
        extract_resume_text(b"x" * (MAX_RESUME_BYTES + 1), "resume.txt")


def test_extract_docx_resume() -> None:
    document = Document()
    document.add_paragraph(RESUME)
    buffer = io.BytesIO()
    document.save(buffer)
    assert "Python" in extract_resume_text(buffer.getvalue(), "resume.docx")


def test_interview_scoring_rewards_star_and_evidence() -> None:
    job = extract_job_requirements(JOB)
    weak = score_interview_answer("介绍项目", "我做过一个项目。", job)
    strong = score_interview_answer(
        "介绍项目",
        "当时团队需要缩短周报时间，我负责数据清洗和分析。首先用 SQL 提取数据，随后用 Python 自动化处理。最终处理 50000 条记录，将时间从 3 小时缩短到 40 分钟。我复盘后把脚本沉淀成模板。",
        job,
    )
    assert strong.total > weak.total
    assert strong.evidence >= weak.evidence


def test_top_three_skills_prioritize_repeated_core_gaps() -> None:
    profile = infer_resume_profile("本科，Excel，内容创作。", education="本科")
    job_a = extract_job_requirements(JOB)
    job_b = JobRequirements(title="商业分析", must_skills=["SQL", "数据分析"], preferred_skills=["Python"])
    analyses = [(job_a, analyze_match(profile, job_a)), (job_b, analyze_match(profile, job_b))]
    priorities = top_skill_priorities(analyses)
    assert priorities[0]["skill"] in {"SQL", "数据分析"}
    assert len(priorities) <= 3
