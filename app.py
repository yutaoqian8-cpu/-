from __future__ import annotations

import html
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ai_service import (
    AIInterviewFeedback,
    AIJobAnalysis,
    analyze_with_ai,
    get_api_config,
    privacy_safe_identifier,
    score_answer_with_ai,
)
from career_core import (
    APP_VERSION,
    EDUCATION_RANK,
    MAX_RESUME_BYTES,
    InterviewScore,
    JobRequirements,
    MatchAnalysis,
    ResumeProfile,
    analysis_to_json,
    analyze_match,
    extract_job_requirements,
    extract_resume_text,
    generate_cover_letter,
    generate_model_answer,
    infer_resume_profile,
    jd_health_check,
    rewrite_bullets,
    role_reference,
    score_interview_answer,
    split_terms,
    top_skill_priorities,
    unique,
)
from knowledge_base import (
    INTERVIEW_QUESTION_BANK,
    ROLE_PLAYBOOK,
    SALARY_REFERENCE,
    find_role_playbook,
    salary_for_key,
)
from storage import (
    STATUSES,
    add_application,
    application_metrics,
    delete_application,
    init_database,
    list_applications,
    update_application_status,
)

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "data" / "careerpilot.db"
LOGGER = logging.getLogger(__name__)

SAMPLE_RESUME = """张同学｜某大学信息管理与信息系统本科｜2026届
技能：Python、SQL、Excel、Pandas、数据分析、数据可视化、沟通协作
项目经历：参与校园消费数据分析项目，负责清洗 5 万条数据，使用 Python 和 SQL 完成指标分析，制作可视化看板，将周报整理时间从 3 小时缩短到 40 分钟。
实习经历：协助互联网产品团队完成用户反馈整理、竞品分析和需求文档撰写，与设计和研发协作推进 2 个功能上线。
证书：大学英语六级。
"""

SAMPLE_JOB = """数据产品实习生
公司：星海科技
岗位职责：
1. 协助分析产品与运营数据，搭建业务指标看板；
2. 参与用户需求分析、竞品研究和产品迭代；
3. 使用 SQL 提取数据，并与产品、研发团队协作推进项目。
任职要求：
1. 本科及以上学历，计算机、统计、信息管理等专业优先；
2. 熟练使用 SQL、Excel，具备数据分析和逻辑分析能力；
3. 熟悉 Python、Pandas 或数据可视化工具者优先；
4. 沟通能力强，每周可实习 4 天，连续 3 个月以上。
"""


def init_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": str(uuid.uuid4()),
        "resume_text": "",
        "job_text": "",
        "current_profile": None,
        "current_job": None,
        "current_analysis": None,
        "analysis_history": [],
        "ai_analysis": None,
        "interview_score": None,
        "api_key_override": "",
        "model_override": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    init_database(DB_PATH)


def inject_css() -> None:
    css = (PROJECT_DIR / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        f"""
        <section class="cp-hero">
          <div class="cp-eyebrow">CAREERPILOT · V{APP_VERSION}</div>
          <h1>把每次投递，<br>变成一次有反馈的训练。</h1>
          <p>从简历与岗位匹配，到面试训练、投递跟踪和跨岗位能力补强。先看证据，再给建议，不虚构经历，也不把匹配信号当成 Offer 概率。</p>
          <div class="cp-hero-tags">
            <span class="cp-hero-tag">简历 × JD 诊断</span>
            <span class="cp-hero-tag">面试模拟评分</span>
            <span class="cp-hero-tag">投递进度管理</span>
            <span class="cp-hero-tag">Top 3 能力雷达</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def load_sample() -> None:
    st.session_state["resume_text"] = SAMPLE_RESUME
    st.session_state["job_text"] = SAMPLE_JOB
    st.session_state["cp_school"] = "某大学"
    st.session_state["cp_education"] = "本科"
    st.session_state["cp_major"] = "信息管理与信息系统"
    st.session_state["cp_graduation"] = "2026"
    st.session_state["cp_targets"] = "数据分析、数据产品"
    st.session_state["cp_skills"] = "Python、SQL、Excel、Pandas、数据分析、数据可视化、沟通协作"
    st.session_state["cp_highlights"] = "清洗5万条数据，周报整理时间从3小时缩短到40分钟"


def _effective_api_config() -> tuple[str, str]:
    api_key, model = get_api_config(st.secrets)
    if st.session_state.get("api_key_override"):
        api_key = st.session_state["api_key_override"]
    if st.session_state.get("model_override"):
        model = st.session_state["model_override"]
    return api_key, model


def render_profile_and_job() -> tuple[ResumeProfile, str]:
    st.markdown('<div class="cp-kicker">01 · 准备材料</div>', unsafe_allow_html=True)
    st.header("把简历与岗位放到同一张作战地图")
    st.caption("简历默认只在当前会话中使用，不写入投递数据库。支持上传或直接粘贴文本。")

    if st.button("✨ 填入虚构示例，一键体验"):
        load_sample()
        st.rerun()

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 你的简历")
            uploaded = st.file_uploader(
                "上传简历",
                type=["pdf", "docx", "txt", "md"],
                help=f"单个文件不超过 {MAX_RESUME_BYTES // 1024 // 1024} MB；扫描版 PDF 请先 OCR。",
            )
            if uploaded is not None:
                try:
                    extracted = extract_resume_text(uploaded.getvalue(), uploaded.name)
                    if extracted != st.session_state.get("resume_text"):
                        st.session_state["resume_text"] = extracted
                    st.success(f"已解析 {uploaded.name}，提取 {len(extracted):,} 个字符。")
                except (ValueError, OSError, ImportError) as exc:
                    st.error(str(exc))

            resume_text = st.text_area(
                "简历文字",
                key="resume_text",
                height=260,
                placeholder="粘贴教育经历、实习、项目、技能和成果；建议删除身份证号、家庭住址等无关敏感信息。",
            )
            with st.expander("补充个人情况（有助于提高规则分析准确性）"):
                c1, c2 = st.columns(2)
                name = c1.text_input("姓名或称呼（选填）", key="cp_name")
                school = c2.text_input("学校", key="cp_school")
                c3, c4 = st.columns(2)
                education = c3.selectbox(
                    "学历", ["请选择", *[key for key in EDUCATION_RANK if key != "不限"]], key="cp_education"
                )
                major = c4.text_input("专业", key="cp_major")
                c5, c6 = st.columns(2)
                graduation = c5.text_input("毕业年份", key="cp_graduation", placeholder="例如：2026")
                years = c6.number_input(
                    "累计相关经验（年）", min_value=0.0, max_value=20.0, step=0.25, key="cp_years"
                )
                targets = st.text_input("目标岗位", key="cp_targets", placeholder="例如：产品经理、数据分析")
                skills = st.text_input("补充技能", key="cp_skills", placeholder="例如：SQL、Figma、用户研究")
                highlights = st.text_area(
                    "可量化成果",
                    key="cp_highlights",
                    height=90,
                    placeholder="例如：将处理时间缩短30%；覆盖2万用户；独立完成3次活动复盘",
                )

    with right:
        with st.container(border=True):
            st.markdown("#### 目标岗位 JD")
            job_text = st.text_area(
                "粘贴招聘岗位",
                key="job_text",
                height=510,
                placeholder="建议粘贴：岗位名称、公司、岗位职责、任职要求、加分项和实习时长。",
            )
            st.caption("岗位文本可能包含广告或指令；系统只把它当作待分析数据。")

    profile = infer_resume_profile(
        resume_text,
        name=name,
        school=school,
        education=education if education != "请选择" else "",
        major=major,
        graduation_year=graduation,
        target_roles=split_terms(targets),
        manual_skills=split_terms(skills),
        experience_years=float(years),
        highlights=split_terms(highlights),
    )
    return profile, job_text


def run_analysis(profile: ResumeProfile, job_text: str) -> None:
    if len(profile.raw_text) < 30 and not profile.skills:
        st.error("请先上传或粘贴简历，并至少提供一段经历或技能。")
        return
    try:
        job = extract_job_requirements(job_text)
        analysis = analyze_match(profile, job)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["current_profile"] = profile
    st.session_state["current_job"] = job
    st.session_state["current_analysis"] = analysis
    st.session_state["ai_analysis"] = None
    history = st.session_state["analysis_history"]
    history.append((job, analysis))
    st.session_state["analysis_history"] = history[-20:]
    st.success("分析完成：已拆解岗位要求、计算匹配信号并生成行动建议。")


def _pills(values: list[str], gap: bool = False) -> None:
    css_class = "cp-pill cp-gap" if gap else "cp-pill"
    if not values:
        st.caption("暂无")
        return
    st.markdown(
        "".join(f'<span class="{css_class}">{html.escape(value)}</span>' for value in values),
        unsafe_allow_html=True,
    )


def render_rule_results(profile: ResumeProfile, job: JobRequirements, analysis: MatchAnalysis) -> None:
    st.divider()
    st.markdown('<div class="cp-kicker">02 · 匹配诊断</div>', unsafe_allow_html=True)
    st.header(f"{job.company} · {job.title}")
    top_left, top_right = st.columns([0.34, 0.66], gap="large")
    with top_left:
        with st.container(border=True):
            st.markdown(
                f'<div class="cp-score">{analysis.score}<small> / 100</small></div>', unsafe_allow_html=True
            )
            st.markdown(f"### {analysis.label}")
            st.caption("这是材料匹配信号，不是获得面试或 Offer 的概率。")
    with top_right:
        metrics = st.columns(4)
        metrics[0].metric("核心技能", f"{analysis.skill_score}%")
        metrics[1].metric("经验", f"{analysis.experience_score}%")
        metrics[2].metric("学历", f"{analysis.education_score}%")
        metrics[3].metric("关键词证据", f"{analysis.keyword_score}%")
        st.markdown("**已匹配能力**")
        _pills(analysis.matched_skills)
        st.markdown("**需要补强**")
        _pills([*analysis.missing_skills, *analysis.preferred_gaps], gap=True)

    tab_req, tab_resume, tab_interview, tab_action, tab_ai = st.tabs(
        ["岗位拆解", "简历改写", "面试题", "行动清单", "AI 深度分析"]
    )
    with tab_req:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 硬性 / 核心要求")
            _pills(job.must_skills)
            st.write(f"学历：{job.education}｜经验：{job.min_years:g} 年")
            for item in analysis.evidence:
                st.markdown(f"- {item}")
        with c2:
            st.markdown("#### 优先项与主要职责")
            _pills(job.preferred_skills)
            for item in job.responsibilities or ["未从文本中识别到清晰的职责句，请人工阅读原文。"]:
                st.markdown(f"- {item}")
        ref = role_reference(job)
        if ref:
            playbook = ref["role"]
            st.divider()
            st.markdown("#### 岗位画像与参考")
            st.markdown(f"**画像**：{playbook['summary']}")
            st.markdown(f"**职业发展路径**：{playbook['career_path']}")
            left_ref, right_ref = st.columns(2)
            with left_ref:
                st.markdown("**典型职责**")
                for item in playbook["responsibilities"]:
                    st.markdown(f"- {item}")
            with right_ref:
                st.markdown("**入门建议**")
                for item in playbook["entry_tips"]:
                    st.markdown(f"- {item}")
            bands = ref["salary"]
            if bands:
                st.markdown("**薪资参考（月薪，方向性参考，需结合城市/行业/公司规模校准）**")
                for band in bands:
                    st.markdown(f"- {band['band']}：一线 {band['tier1']} ｜ 二线 {band['tier2']}")

    with tab_resume:
        st.markdown("#### 逐条修改建议")
        for index, item in enumerate(analysis.resume_suggestions, 1):
            st.markdown(f"**{index}.** {item}")
        st.warning("只改写真实发生过的经历。缺失能力应先学习并产出作品，再写进简历。")
        st.markdown("#### STAR 改写草稿")
        for index, item in enumerate(rewrite_bullets(profile, job, analysis.matched_skills), 1):
            st.markdown(f"**{index}.** {item}")
        st.markdown("#### 90 秒自我介绍草稿")
        st.text_area("可复制后继续修改", value=analysis.pitch, height=150, key="pitch_output")
        st.markdown("#### 求职信草稿")
        cover = generate_cover_letter(profile, job, analysis.matched_skills, analysis.missing_skills)
        st.text_area("可复制后继续修改", value=cover, height=320, key="cover_output")

    with tab_interview:
        for index, question in enumerate(analysis.interview_questions, 1):
            st.markdown(f"**Q{index}.** {question}")
        st.caption("可前往“面试训练”模块逐题作答并评分。")
        st.divider()
        st.markdown("#### 示范回答骨架")
        demo_q = st.selectbox("选择题目查看 STAR 示范", analysis.interview_questions, key="demo_question")
        st.text_area(
            "STAR 示范（请替换为你的真实经历）",
            value=generate_model_answer(demo_q, profile, job),
            height=260,
            key="model_answer_output",
        )

    with tab_action:
        for index, action in enumerate(analysis.next_actions, 1):
            st.checkbox(action, key=f"action_{index}_{job.title}_{analysis.score}")
        st.divider()
        st.markdown("#### JD 质量体检")
        health = jd_health_check(job.source_text)
        if health["red"]:
            for item in health["red"]:
                st.warning(item)
        else:
            st.success("未命中常见风险信号（仍建议自行核实公司资质与薪资范围）。")
        with st.expander("投递前可对照核验的加分项"):
            for item in health["green"]:
                st.markdown(f"- {item}")
        report = analysis_to_json(profile, job, analysis)
        st.download_button(
            "下载本次分析 JSON",
            data=report.encode("utf-8"),
            file_name="CareerPilot_岗位分析.json",
            mime="application/json",
            width="stretch",
        )

    with tab_ai:
        api_key, model = _effective_api_config()
        if not api_key:
            st.info("尚未配置 OpenAI API Key。规则分析、面试基础评分和投递看板均可正常使用。")
        st.caption(f"当前模型：{model}。AI 结果是对规则分析的补充，不覆盖可解释的本地分数。")
        consent = st.checkbox(
            "我同意将当前简历文本、岗位描述和规则分析发送给配置的 OpenAI API",
            key="ai_analysis_consent",
        )
        if st.button("生成 AI 深度分析", disabled=not (api_key and consent), width="stretch"):
            try:
                with st.spinner("正在生成结构化深度分析……"):
                    result = analyze_with_ai(
                        profile=profile,
                        job=job,
                        rule_analysis=analysis,
                        api_key=api_key,
                        model=model,
                        safety_identifier=privacy_safe_identifier(st.session_state["session_id"]),
                    )
                st.session_state["ai_analysis"] = result
            except Exception:  # noqa: BLE001 - SDK has multiple transport and validation errors.
                LOGGER.exception("AI analysis failed")
                st.error("AI 分析暂时失败。系统未展示底层错误，请检查模型权限、Key 或网络后重试。")
        ai_result: AIJobAnalysis | None = st.session_state.get("ai_analysis")
        if ai_result:
            st.markdown("#### AI 提炼的岗位要求")
            for item in ai_result.requirement_summary:
                st.markdown(f"- {item}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 匹配证据")
                for item in ai_result.matched_evidence:
                    st.markdown(f"- {item}")
                st.markdown("#### 简历改写建议")
                for item in ai_result.resume_rewrites:
                    st.markdown(f"- {item}")
            with c2:
                st.markdown("#### 能力差距")
                for item in ai_result.capability_gaps:
                    st.markdown(f"- {item}")
                st.markdown("#### 30 天计划")
                for item in ai_result.thirty_day_plan:
                    st.markdown(f"- {item}")
            st.markdown("#### STAR 改写要点")
            for item in ai_result.rewritten_bullets:
                st.markdown(f"- {item}")
            st.markdown("#### AI 补充面试题")
            for item in ai_result.interview_questions:
                st.markdown(f"- {item}")
            st.markdown("#### AI 求职信草稿")
            st.text_area("可复制后继续修改", value=ai_result.cover_letter, height=280, key="ai_cover_output")
            if ai_result.caveats:
                st.markdown("#### 注意事项")
                for item in ai_result.caveats:
                    st.caption(f"- {item}")


def render_analysis_page() -> None:
    profile, job_text = render_profile_and_job()
    if st.button("开始完整分析 →", type="primary", width="stretch"):
        run_analysis(profile, job_text)
    current_profile: ResumeProfile | None = st.session_state.get("current_profile")
    job: JobRequirements | None = st.session_state.get("current_job")
    analysis: MatchAnalysis | None = st.session_state.get("current_analysis")
    if current_profile and job and analysis:
        render_rule_results(current_profile, job, analysis)


def _render_score(score: InterviewScore | AIInterviewFeedback) -> None:
    values = st.columns(5)
    total = score.total if isinstance(score, InterviewScore) else score.total_score
    values[0].metric("总分", total)
    values[1].metric("结构", score.structure if isinstance(score, InterviewScore) else score.structure_score)
    values[2].metric(
        "相关性", score.relevance if isinstance(score, InterviewScore) else score.relevance_score
    )
    values[3].metric("证据", score.evidence if isinstance(score, InterviewScore) else score.evidence_score)
    values[4].metric("清晰度", score.clarity if isinstance(score, InterviewScore) else score.clarity_score)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 做得好的地方")
        for item in score.strengths:
            st.markdown(f"- {item}")
    with right:
        st.markdown("#### 下一轮怎么改")
        for item in score.improvements:
            st.markdown(f"- {item}")
    st.markdown("#### 更好的回答骨架")
    for index, item in enumerate(score.improved_answer_outline, 1):
        st.markdown(f"{index}. {item}")


def render_interview_page() -> None:
    st.markdown('<div class="cp-kicker">03 · 面试训练</div>', unsafe_allow_html=True)
    st.header("把“我会”练成有证据的回答")
    job: JobRequirements | None = st.session_state.get("current_job")
    analysis: MatchAnalysis | None = st.session_state.get("current_analysis")
    if not job or not analysis:
        st.info("请先在“匹配分析”中完成一次岗位分析，系统会自动生成对应问题。")
        return

    with st.container(border=True):
        question = st.selectbox("选择一道题", analysis.interview_questions)
        answer = st.text_area(
            "你的回答",
            height=240,
            placeholder="建议使用 STAR：背景、任务、你的具体行动、量化结果与复盘。",
        )
        use_ai = st.toggle("使用 AI 教练评分", value=False)
        api_key, model = _effective_api_config()
        consent = True
        if use_ai:
            consent = st.checkbox("我同意将本题、回答和岗位摘要发送给配置的 OpenAI API")
            if not api_key:
                st.warning("未配置 API Key，将使用本地评分。")
        if st.button("提交回答并评分", type="primary", width="stretch", disabled=not answer.strip()):
            local_score = score_interview_answer(question, answer, job)
            final_score: InterviewScore | AIInterviewFeedback = local_score
            if use_ai and api_key and consent:
                try:
                    with st.spinner("AI 教练正在复盘……"):
                        final_score = score_answer_with_ai(
                            question=question,
                            answer=answer,
                            job=job,
                            local_score=local_score,
                            api_key=api_key,
                            model=model,
                            safety_identifier=privacy_safe_identifier(st.session_state["session_id"]),
                        )
                except Exception:  # noqa: BLE001
                    LOGGER.exception("AI interview scoring failed")
                    st.warning("AI 评分暂时不可用，已展示本地评分结果。")
            st.session_state["interview_score"] = final_score
    score = st.session_state.get("interview_score")
    if score:
        _render_score(score)


def render_tracker_page() -> None:
    st.markdown('<div class="cp-kicker">04 · 投递看板</div>', unsafe_allow_html=True)
    st.header("别让投递记录散落在聊天和收藏夹里")
    frame = list_applications(DB_PATH)
    metrics = application_metrics(frame)
    cols = st.columns(5)
    cols[0].metric("累计投递", metrics["total"])
    cols[1].metric("进行中", metrics["active"])
    cols[2].metric("面试中", metrics["interviews"])
    cols[3].metric("Offer", metrics["offers"])
    cols[4].metric("反馈率", f"{metrics['response_rate']}%")

    job: JobRequirements | None = st.session_state.get("current_job")
    analysis: MatchAnalysis | None = st.session_state.get("current_analysis")
    with st.expander("新增一条投递记录", expanded=frame.empty):
        with st.form("add_application_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            company = c1.text_input("公司 *", value=job.company if job else "")
            role = c2.text_input("岗位 *", value=job.title if job else "")
            c3, c4 = st.columns(2)
            applied_date = c3.date_input("日期", value=date.today())
            status = c4.selectbox("当前进度", STATUSES, index=0)
            url = st.text_input("招聘链接（选填）")
            next_action = st.text_input("下一步", placeholder="例如：8月8日跟进 HR；准备 SQL 笔试")
            notes = st.text_area("备注", height=80)
            submitted = st.form_submit_button("保存到本地看板", type="primary", width="stretch")
            if submitted:
                if not company.strip() or not role.strip():
                    st.error("公司和岗位为必填项。")
                else:
                    skills = unique(
                        [*(job.must_skills if job else []), *(job.preferred_skills if job else [])]
                    )
                    add_application(
                        DB_PATH,
                        company=company,
                        role=role,
                        applied_date=applied_date,
                        status=status,
                        score=analysis.score if analysis else None,
                        skills=skills,
                        url=url,
                        next_action=next_action,
                        notes=notes,
                    )
                    st.success("已保存。")
                    st.rerun()

    if frame.empty:
        st.info("暂无投递记录。完成一次岗位分析后，可以一键带入公司、岗位和匹配信号。")
        return
    st.dataframe(frame, width="stretch", hide_index=True, height=390)
    st.download_button(
        "导出投递记录 CSV",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="CareerPilot_投递记录.csv",
        mime="text/csv",
    )
    with st.expander("更新或删除记录"):
        ids = frame["id"].astype(int).tolist()
        selected_id = st.selectbox("记录 ID", ids)
        new_status = st.selectbox("更新进度", STATUSES, key="update_status")
        new_action = st.text_input("更新下一步")
        c1, c2 = st.columns(2)
        if c1.button("保存更新", width="stretch"):
            update_application_status(DB_PATH, int(selected_id), new_status, new_action)
            st.success("状态已更新。")
            st.rerun()
        confirm_delete = st.checkbox("确认删除所选记录")
        if c2.button("删除记录", width="stretch", disabled=not confirm_delete):
            delete_application(DB_PATH, int(selected_id))
            st.success("记录已删除。")
            st.rerun()


def render_skill_radar_page() -> None:
    st.markdown('<div class="cp-kicker">05 · 能力雷达</div>', unsafe_allow_html=True)
    st.header("多个岗位都在要什么，才值得优先补什么")
    history: list[tuple[JobRequirements, MatchAnalysis]] = st.session_state.get("analysis_history", [])
    if not history:
        st.info("至少完成一次岗位分析后，这里会汇总高频能力差距。建议连续分析 3—5 个真实岗位。")
        return

    priorities = top_skill_priorities(history)
    if not priorities:
        st.success("当前分析的岗位没有识别到明显重复能力差距。可以增加更多目标岗位验证。")
    else:
        cols = st.columns(len(priorities))
        for index, (column, item) in enumerate(zip(cols, priorities, strict=False), 1):
            with column:
                with st.container(border=True):
                    st.markdown(f"### #{index} {item['skill']}")
                    st.caption(f"优先级信号 {item['priority']} · 涉及岗位：{'、'.join(item['roles'])}")
                    for plan in item["plan"]:
                        st.markdown(f"- {plan}")

    rows = []
    for job, analysis in history:
        rows.append(
            {
                "岗位": job.title,
                "匹配信号": analysis.score,
                "已匹配技能数": len(analysis.matched_skills),
                "核心缺口数": len(analysis.missing_skills),
            }
        )
    summary = pd.DataFrame(rows).drop_duplicates(subset=["岗位"], keep="last")
    st.markdown("#### 岗位组合概览")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.bar_chart(summary.set_index("岗位")[["匹配信号"]], color="#3157D5")


def render_toolbox_page() -> None:
    st.markdown('<div class="cp-kicker">06 · 求职工具箱</div>', unsafe_allow_html=True)
    st.header("岗位画像、薪资、面试题与求职材料，一站查")
    st.caption("以下为参考信息，用于拓宽认知与准备；具体以目标岗位 JD 与招聘平台实时数据为准。")

    tab_role, tab_salary, tab_questions, tab_docs, tab_jd = st.tabs(
        ["岗位参考库", "薪资参考", "面试题库", "求职材料", "JD 体检"]
    )

    with tab_role:
        role_names = list(ROLE_PLAYBOOK.keys())
        selected_role = st.selectbox("选择岗位", role_names)
        playbook = ROLE_PLAYBOOK[selected_role]
        st.markdown(f"### {selected_role}")
        st.markdown(f"**画像**：{playbook['summary']}")
        st.markdown(f"**职业发展路径**：{playbook['career_path']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**核心职责**")
            for item in playbook["responsibilities"]:
                st.markdown(f"- {item}")
            st.markdown("**必备技能**")
            _pills(playbook["must_skills"])
            st.markdown("**加分技能**")
            _pills(playbook["preferred_skills"], gap=True)
        with c2:
            st.markdown("**常见面试题**")
            for item in playbook["interview_questions"]:
                st.markdown(f"- {item}")
            st.markdown("**入门建议**")
            for item in playbook["entry_tips"]:
                st.markdown(f"- {item}")

    with tab_salary:
        role_for_salary = st.selectbox("选择岗位", list(SALARY_REFERENCE.keys()), key="salary_role")
        tier = st.radio("城市等级", ["一线", "二线"], horizontal=True)
        bands = salary_for_key(role_for_salary) or []
        st.markdown("### 薪资参考区间（月薪）")
        for band in bands:
            value = band["tier1"] if tier == "一线" else band["tier2"]
            st.markdown(f"- **{band['band']}**：{value}")
        st.caption("以上为方向性参考，随城市、行业、公司规模与个人能力浮动，投递前以招聘平台实时数据为准。")

    with tab_questions:
        category = st.selectbox("问题类型", list(INTERVIEW_QUESTION_BANK.keys()))
        for question in INTERVIEW_QUESTION_BANK[category]:
            st.markdown(f"- {question}")
        st.caption("把每题都准备成 STAR 结构，面试时更从容。")

    with tab_docs:
        profile = st.session_state.get("current_profile")
        job = st.session_state.get("current_job")
        analysis = st.session_state.get("current_analysis")
        if not profile or not job or not analysis:
            st.info("请先在「匹配分析」完成一次分析，即可在这里生成求职信与简历改写。")
        else:
            st.markdown("#### STAR 简历改写草稿")
            for index, item in enumerate(rewrite_bullets(profile, job, analysis.matched_skills), 1):
                st.markdown(f"**{index}.** {item}")
            st.markdown("#### 求职信草稿")
            st.text_area(
                "可复制后继续修改",
                value=generate_cover_letter(profile, job, analysis.matched_skills, analysis.missing_skills),
                height=320,
                key="toolbox_cover",
            )
            st.markdown("#### 90 秒自我介绍")
            st.text_area("可复制后继续修改", value=analysis.pitch, height=150, key="toolbox_pitch")

    with tab_jd:
        jd_text = st.text_area("粘贴岗位 JD 进行体检", height=260, key="toolbox_jd")
        if st.button("开始体检", width="stretch"):
            if len(jd_text.strip()) < 30:
                st.warning("JD 过短，请粘贴完整内容。")
            else:
                health = jd_health_check(jd_text)
                if health["red"]:
                    st.markdown("#### 风险信号")
                    for item in health["red"]:
                        st.warning(item)
                else:
                    st.success("未命中常见风险信号（仍需自行核实公司资质与薪资范围）。")
                with st.expander("投递前可对照核验的加分项"):
                    for item in health["green"]:
                        st.markdown(f"- {item}")


def render_settings_page() -> None:
    st.markdown('<div class="cp-kicker">设置 · 隐私</div>', unsafe_allow_html=True)
    st.header("AI 是增强项，不是产品的单点依赖")
    api_key, model = _effective_api_config()
    with st.container(border=True):
        st.markdown("#### OpenAI 配置")
        st.write("状态：" + ("✅ 已配置 API Key" if api_key else "○ 未配置 API Key"))
        st.text_input(
            "会话级 API Key（不会写入数据库）",
            type="password",
            key="api_key_override",
            placeholder="也可以通过 .streamlit/secrets.toml 配置",
        )
        st.text_input("模型", key="model_override", placeholder=model)
        st.caption("默认建议使用 gpt-5.6-terra 平衡质量与成本；可按账户可用模型自行修改。")

    with st.container(border=True):
        st.markdown("#### 数据边界")
        st.markdown(
            """
            - 简历原文保存在当前 Streamlit 会话，不写入投递数据库。
            - 投递记录保存在本机 `data/careerpilot.db`。
            - 只有勾选明确同意并点击 AI 按钮时，相关文本才会发送给 OpenAI API。
            - API 请求设置为 `store=False`；简历导出报告默认移除完整原文。
            - 建议上传前删除身份证号、详细住址、银行卡等与求职分析无关的信息。
            """
        )


def main() -> None:
    st.set_page_config(page_title="CareerPilot｜实习求职作战智能体", page_icon="🧭", layout="wide")
    init_state()
    inject_css()
    render_header()

    page = st.segmented_control(
        "功能导航",
        ["匹配分析", "面试训练", "求职工具箱", "投递看板", "能力雷达", "设置与隐私"],
        default="匹配分析",
        selection_mode="single",
        label_visibility="collapsed",
    )
    st.divider()
    if page == "面试训练":
        render_interview_page()
    elif page == "求职工具箱":
        render_toolbox_page()
    elif page == "投递看板":
        render_tracker_page()
    elif page == "能力雷达":
        render_skill_radar_page()
    elif page == "设置与隐私":
        render_settings_page()
    else:
        render_analysis_page()


if __name__ == "__main__":
    main()
