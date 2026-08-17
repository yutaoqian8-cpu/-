from __future__ import annotations

import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from knowledge_base import (
    EXTRA_SKILL_TAXONOMY,
    INTERVIEW_QUESTION_BANK,
    JD_GREEN_FLAGS,
    JD_RED_FLAGS,
    LEARNING_PATHS,
    find_role_playbook,
    learning_path_for,
    salary_for_role,
    skill_effort,
)

APP_VERSION = "1.1"
MAX_RESUME_BYTES = 8 * 1024 * 1024

EDUCATION_RANK = {"不限": 0, "大专": 1, "本科": 2, "硕士": 3, "博士": 4}

SKILL_TAXONOMY: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "Java": ("java",),
    "C++": ("c++", "cpp"),
    "JavaScript": ("javascript", "js"),
    "TypeScript": ("typescript", "ts"),
    "React": ("react", "react.js"),
    "Vue": ("vue", "vue.js"),
    "SQL": ("sql", "mysql", "postgresql", "数据库查询"),
    "Excel": ("excel", "vlookup", "数据透视表", "power query"),
    "Pandas": ("pandas",),
    "数据分析": ("数据分析", "商业分析", "经营分析"),
    "数据可视化": ("数据可视化", "tableau", "power bi", "matplotlib"),
    "机器学习": ("机器学习", "machine learning", "sklearn", "scikit-learn"),
    "深度学习": ("深度学习", "pytorch", "tensorflow"),
    "大语言模型": ("大语言模型", "llm", "生成式ai", "生成式 ai", "prompt"),
    "产品设计": ("产品设计", "原型设计", "figma", "axure"),
    "产品运营": ("产品运营", "用户运营", "内容运营", "活动运营"),
    "用户研究": ("用户研究", "用户访谈", "问卷", "可用性测试"),
    "需求分析": ("需求分析", "需求文档", "prd", "用户故事"),
    "项目管理": ("项目管理", "项目推进", "敏捷", "scrum"),
    "市场研究": ("市场研究", "行业研究", "竞品分析", "市场分析"),
    "新媒体运营": ("新媒体", "公众号", "小红书", "抖音运营"),
    "内容创作": ("内容创作", "文案", "脚本撰写", "编辑"),
    "SEO": ("seo", "搜索引擎优化"),
    "SEM": ("sem", "搜索引擎营销"),
    "英语": ("英语", "cet-4", "cet4", "cet-6", "cet6", "雅思", "托福"),
    "沟通协作": ("沟通协作", "跨部门沟通", "团队协作", "沟通能力"),
    "逻辑分析": ("逻辑分析", "结构化思维", "分析能力"),
    "演讲表达": ("演讲", "汇报", "表达能力", "presentation"),
    "财务分析": ("财务分析", "财务建模", "估值", "三大报表"),
    "会计": ("会计", "财务核算", "记账"),
    "招聘": ("招聘", "人才寻访", "简历筛选", "面试邀约"),
    "人力资源": ("人力资源", "hr", "员工关系", "绩效管理"),
    "客户成功": ("客户成功", "客户运营", "客户维护"),
    "销售": ("销售", "商务拓展", "bd", "客户开发"),
    "Git": ("git", "github", "gitlab"),
    "Linux": ("linux", "shell"),
    "Docker": ("docker", "容器化"),
    "云计算": ("云计算", "aws", "azure", "阿里云", "腾讯云"),
}
# 合并扩充技能词表，覆盖更多岗位与场景。
SKILL_TAXONOMY.update(EXTRA_SKILL_TAXONOMY)

PREFERRED_MARKERS = ("优先", "加分", "最好", "熟悉更佳", "有经验者", "plus")
MUST_MARKERS = ("必须", "要求", "具备", "熟练", "掌握", "能够", "负责", "任职资格")
STOPWORDS = {
    "负责",
    "工作",
    "相关",
    "岗位",
    "能力",
    "要求",
    "进行",
    "以及",
    "具有",
    "具备",
    "优先",
    "熟悉",
    "公司",
    "团队",
    "完成",
    "以上",
}


@dataclass
class ResumeProfile:
    name: str = ""
    school: str = ""
    education: str = ""
    major: str = ""
    graduation_year: str = ""
    target_roles: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    highlights: list[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class JobRequirements:
    title: str = "目标岗位"
    company: str = "目标公司"
    must_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    education: str = "不限"
    min_years: float = 0.0
    keywords: list[str] = field(default_factory=list)
    source_text: str = ""


@dataclass
class MatchAnalysis:
    score: int
    label: str
    skill_score: int
    experience_score: int
    education_score: int
    keyword_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    preferred_gaps: list[str]
    evidence: list[str]
    resume_suggestions: list[str]
    interview_questions: list[str]
    pitch: str
    next_actions: list[str]


@dataclass
class InterviewScore:
    total: int
    structure: int
    relevance: int
    evidence: int
    clarity: int
    strengths: list[str]
    improvements: list[str]
    improved_answer_outline: list[str]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\r\n?", "\n", str(value)).strip()


def compact(value: str) -> str:
    return re.sub(r"[\s\u3000,，;；、/（）()【】\[\]：:·\-—_]+", "", value).lower()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def split_terms(value: str) -> list[str]:
    return unique(re.split(r"[,，;；、/\n]+", value or ""))


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    if not file_bytes:
        raise ValueError("简历文件为空。")
    if len(file_bytes) > MAX_RESUME_BYTES:
        raise ValueError("简历文件超过 8 MB，请压缩或精简后重试。")

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        if len(reader.pages) > 30:
            raise ValueError("简历 PDF 超过 30 页，请上传精简版本。")
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document

        document = Document(io.BytesIO(file_bytes))
        content = "\n".join(paragraph.text for paragraph in document.paragraphs)
    elif suffix in {".txt", ".md"}:
        content = ""
        for encoding in ("utf-8-sig", "gb18030", "utf-8"):
            try:
                content = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not content:
            raise ValueError("无法识别文本文件编码。")
    else:
        raise ValueError("仅支持 PDF、DOCX、TXT 和 Markdown 简历。")

    content = clean_text(content)
    if len(content) < 30:
        raise ValueError("未能从简历中提取足够文字；扫描版 PDF 请先进行 OCR。")
    return content[:80_000]


def _contains_alias(text_value: str, alias: str) -> bool:
    if re.fullmatch(r"[a-zA-Z0-9+#.\- ]+", alias):
        return (
            re.search(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])", text_value.lower()) is not None
        )
    return compact(alias) in compact(text_value)


def extract_skills(text_value: str) -> list[str]:
    found: list[str] = []
    for canonical, aliases in SKILL_TAXONOMY.items():
        if any(_contains_alias(text_value, alias) for alias in aliases):
            found.append(canonical)
    return found


def infer_education(text_value: str) -> str:
    hits = [level for level in ("博士", "硕士", "本科", "大专") if level in text_value]
    return max(hits, key=lambda item: EDUCATION_RANK[item], default="")


def infer_experience_years(text_value: str) -> float:
    candidates = [
        float(value)
        for value in re.findall(r"(\d+(?:\.\d+)?)\s*年(?:以上)?(?:工作|实习|项目)?经验", text_value)
    ]
    return max(candidates, default=0.0)


def infer_resume_profile(
    raw_text: str,
    *,
    name: str = "",
    school: str = "",
    education: str = "",
    major: str = "",
    graduation_year: str = "",
    target_roles: list[str] | None = None,
    manual_skills: list[str] | None = None,
    experience_years: float | None = None,
    highlights: list[str] | None = None,
) -> ResumeProfile:
    inferred_skills = extract_skills(raw_text)
    return ResumeProfile(
        name=name.strip(),
        school=school.strip(),
        education=education if education in EDUCATION_RANK else infer_education(raw_text),
        major=major.strip(),
        graduation_year=graduation_year.strip(),
        target_roles=unique(target_roles or []),
        skills=unique([*inferred_skills, *(manual_skills or [])]),
        experience_years=float(
            experience_years if experience_years is not None else infer_experience_years(raw_text)
        ),
        highlights=unique(highlights or []),
        raw_text=raw_text[:80_000],
    )


def _extract_title_company(job_text: str) -> tuple[str, str]:
    lines = [line.strip(" -•\t") for line in job_text.splitlines() if line.strip()]
    title, company = "目标岗位", "目标公司"
    for line in lines[:8]:
        title_match = re.search(r"(?:岗位|职位|招聘职位|position)\s*[：:]\s*(.{2,40})", line, flags=re.I)
        company_match = re.search(r"(?:公司|企业|单位|company)\s*[：:]\s*(.{2,50})", line, flags=re.I)
        if title_match:
            title = title_match.group(1).strip()
        if company_match:
            company = company_match.group(1).strip()
    if title == "目标岗位" and lines:
        first = lines[0]
        if len(first) <= 35 and not any(marker in first for marker in MUST_MARKERS):
            title = first
    return title, company


def _extract_education_requirement(job_text: str) -> str:
    minimum_patterns = (
        (r"博士(?:学历|研究生)?(?:及以上|以上)", "博士"),
        (r"硕士(?:研究生)?(?:学历)?(?:及以上|以上)", "硕士"),
        (r"本科(?:学历)?(?:及以上|以上)", "本科"),
        (r"(?:大专|专科)(?:学历)?(?:及以上|以上)", "大专"),
    )
    for pattern, level in minimum_patterns:
        if re.search(pattern, job_text):
            return level
    for level in ("博士", "硕士", "本科", "大专"):
        if level in job_text:
            return level
    return "不限"


def _extract_min_years(job_text: str) -> float:
    patterns = (
        r"(\d+(?:\.\d+)?)\s*年(?:及以上|以上)?(?:相关)?(?:工作|实习|项目)?经验",
        r"(?:工作|实习|项目)经验(?:不少于|至少|满)?\s*(\d+(?:\.\d+)?)\s*年",
    )
    for pattern in patterns:
        match = re.search(pattern, job_text)
        if match:
            return float(match.group(1))
    return 0.0


def _important_keywords(job_text: str, limit: int = 18) -> list[str]:
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", job_text)
    english = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,20}", job_text)]
    counts = Counter(token for token in [*chinese, *english] if token.lower() not in STOPWORDS)
    skill_words = extract_skills(job_text)
    ranked = [word for word, _ in counts.most_common(limit)]
    return unique([*skill_words, *ranked])[:limit]


def extract_job_requirements(job_text: str) -> JobRequirements:
    value = clean_text(job_text)
    if len(value) < 30:
        raise ValueError("岗位描述过短，请粘贴完整的岗位职责和任职要求。")
    if len(value) > 80_000:
        raise ValueError("岗位描述超过 80,000 字，请精简后重试。")

    title, company = _extract_title_company(value)
    lines = [line.strip(" -•\t0123456789.、）)") for line in value.splitlines() if line.strip()]
    all_skills = extract_skills(value)
    preferred: list[str] = []
    clauses = [clause.strip() for clause in re.split(r"[。；;\n]", value) if clause.strip()]
    for clause in clauses:
        if any(marker in clause.lower() for marker in PREFERRED_MARKERS):
            preferred.extend(extract_skills(clause))
    preferred = unique(preferred)
    must = [skill for skill in all_skills if skill not in preferred]

    responsibilities = [
        line
        for line in lines
        if 8 <= len(line) <= 120
        and not line.lower().startswith(("岗位：", "职位：", "公司：", "任职要求：", "岗位要求："))
        and any(marker in line for marker in ("负责", "协助", "参与", "推进", "完成", "支持", "分析", "维护"))
    ][:8]

    return JobRequirements(
        title=title,
        company=company,
        must_skills=must,
        preferred_skills=preferred,
        responsibilities=unique(responsibilities),
        education=_extract_education_requirement(value),
        min_years=_extract_min_years(value),
        keywords=_important_keywords(value),
        source_text=value,
    )


def _education_score(candidate: str, requirement: str) -> int:
    if requirement == "不限":
        return 100
    if candidate not in EDUCATION_RANK:
        return 40
    return 100 if EDUCATION_RANK[candidate] >= EDUCATION_RANK[requirement] else 0


def _experience_score(candidate_years: float, requirement_years: float) -> int:
    if requirement_years <= 0:
        return 100
    return min(100, round(100 * candidate_years / requirement_years))


def _keyword_score(profile: ResumeProfile, job: JobRequirements) -> int:
    if not job.keywords:
        return 100
    haystack = compact(" ".join([profile.raw_text, *profile.skills, *profile.highlights]))
    hits = sum(compact(keyword) in haystack for keyword in job.keywords)
    return round(100 * hits / len(job.keywords))


def generate_resume_suggestions(
    profile: ResumeProfile,
    job: JobRequirements,
    matched: list[str],
    missing: list[str],
) -> list[str]:
    suggestions = [
        f"把简历标题改成“{job.title}候选人｜{profile.major or '专业背景'}｜{matched[0] if matched else '核心能力'}”，让定位在首屏可见。",
        "每段经历使用“动作 + 任务 + 方法 + 量化结果”结构；没有数字时可写覆盖范围、协作人数或交付周期。",
        "将与岗位无关的课程和校园描述压缩，把最相关的项目或实习移到教育经历之前。",
    ]
    if matched:
        suggestions.append(
            f"在项目与实习中补充这些已具备能力的证据：{'、'.join(matched[:5])}；不要只放在技能清单。"
        )
    if missing:
        suggestions.append(
            f"不要虚构缺失技能。优先补齐{'、'.join(missing[:3])}，完成作品或课程后再写入简历。"
        )
    if not profile.highlights:
        suggestions.append("补充 2—3 条可验证成果，例如转化提升、效率提升、阅读量、准确率或项目交付结果。")
    return suggestions[:6]


def generate_interview_questions(profile: ResumeProfile, job: JobRequirements, gaps: list[str]) -> list[str]:
    questions = [
        f"请用 90 秒介绍自己，并说明为什么适合{job.title}。",
        "选一个最能代表你的项目，用 STAR 结构说明目标、你的具体行动和量化结果。",
        "讲一次你在信息不完整或时间紧张时推进任务的经历，你如何做取舍？",
    ]
    playbook = find_role_playbook(f"{job.title} {' '.join(job.responsibilities)}")
    if playbook:
        questions.extend(playbook["interview_questions"][:4])
    questions.extend(INTERVIEW_QUESTION_BANK["行为面试(STAR)"][:2])
    questions.extend(INTERVIEW_QUESTION_BANK["情景/压力题"][:1])
    questions.extend(f"岗位强调{skill}，请说明你的理解，并给出一个真实使用场景。" for skill in gaps[:3])
    if job.responsibilities:
        questions.append(f"针对“{job.responsibilities[0]}”，你会如何制定第一周的行动计划？")
    questions.append("你对我们公司或这个岗位还有什么想了解的吗？")
    return unique(questions)[:12]


def generate_pitch(profile: ResumeProfile, job: JobRequirements, matched: list[str]) -> str:
    identity = (
        "、".join(value for value in [profile.school, profile.major, profile.education] if value)
        or "相关专业背景"
    )
    ability = "、".join(matched[:3]) or "快速学习、沟通协作"
    proof = (
        profile.highlights[0] if profile.highlights else "曾在课程、项目或实习中完成过从分析到交付的完整任务"
    )
    return (
        f"您好，我是{profile.name or '候选人'}，具备{identity}。我与{job.title}最相关的能力是{ability}，"
        f"{proof}。我希望把这些经验用于岗位的核心任务，也会优先补齐当前能力差距。"
    )


def analyze_match(profile: ResumeProfile, job: JobRequirements) -> MatchAnalysis:
    resume_skills = set(profile.skills)
    must = set(job.must_skills)
    preferred = set(job.preferred_skills)
    matched = sorted(resume_skills & (must | preferred))
    missing = sorted(must - resume_skills)
    preferred_gaps = sorted(preferred - resume_skills)

    if must:
        skill_score = round(100 * len(resume_skills & must) / len(must))
    elif preferred:
        skill_score = round(100 * len(resume_skills & preferred) / len(preferred))
    else:
        skill_score = 60
    experience_score = _experience_score(profile.experience_years, job.min_years)
    education_score = _education_score(profile.education, job.education)
    keyword_score = _keyword_score(profile, job)
    score = round(skill_score * 0.6 + experience_score * 0.15 + education_score * 0.1 + keyword_score * 0.15)
    if education_score == 0:
        score = min(score, 59)
    if job.min_years > 0 and experience_score < 50:
        score = min(score, 69)
    label = (
        "重点投递"
        if score >= 80
        else "可以尝试"
        if score >= 65
        else "补强后投递"
        if score >= 45
        else "暂不优先"
    )

    evidence = []
    if matched:
        evidence.append(f"已覆盖核心技能：{'、'.join(matched[:6])}")
    if job.education != "不限":
        gap = (
            "已达标"
            if profile.education and _education_score(profile.education, job.education) == 100
            else "未达标或未填写"
        )
        evidence.append(f"学历要求：{job.education}（当前：{profile.education or '未填写'}）→ {gap}")
    if job.min_years:
        evidence.append(
            f"经验要求：约 {job.min_years:g} 年（当前：{profile.experience_years:g} 年）→ "
            f"{'达标' if experience_score == 100 else '有差距'}"
        )
    if not job.must_skills:
        evidence.append("岗位描述中没有识别到明确技能词，技能项按保守分值处理")

    # 补充缺口优先级（快速 / 中等 / 较长）
    if missing:
        ranked = sorted(missing, key=lambda s: {"快速": 0, "中等": 1, "较长": 2}.get(skill_effort(s), 1))
        evidence.append(
            "技能缺口优先级（按补齐成本）："
            + "；".join(f"{skill}（{skill_effort(skill)}）" for skill in ranked[:5])
        )
    playbook = find_role_playbook(f"{job.title} {' '.join(job.responsibilities)}")
    if playbook:
        evidence.append(
            f"岗位画像：{playbook['summary']} 发展路径：{playbook['career_path']}"
        )

    next_actions = []
    if missing:
        next_actions.append(f"优先补齐{'、'.join(missing[:3])}（先做一个可验证的小项目）")
    next_actions.extend(
        [
            "按岗位关键词重排简历，只保留可验证的能力证据",
            "准备 3 个 STAR 故事：成果、协作、失败复盘",
            "把缺失技能写进 30 天学习计划并产出作品",
            "投递后 3—5 个工作日记录反馈并决定是否跟进",
        ]
    )
    return MatchAnalysis(
        score=score,
        label=label,
        skill_score=skill_score,
        experience_score=experience_score,
        education_score=education_score,
        keyword_score=keyword_score,
        matched_skills=matched,
        missing_skills=missing,
        preferred_gaps=preferred_gaps,
        evidence=evidence,
        resume_suggestions=generate_resume_suggestions(profile, job, matched, missing),
        interview_questions=generate_interview_questions(profile, job, missing),
        pitch=generate_pitch(profile, job, matched),
        next_actions=next_actions[:5],
    )


def score_interview_answer(question: str, answer: str, job: JobRequirements) -> InterviewScore:
    value = clean_text(answer)
    if len(value) < 20:
        return InterviewScore(
            total=20,
            structure=20,
            relevance=30,
            evidence=10,
            clarity=20,
            strengths=[],
            improvements=["回答过短，至少说明背景、你的行动和结果。", "避免只给结论，补充可验证细节。"],
            improved_answer_outline=[
                "一句话给出结论",
                "交代背景和目标",
                "说明你的具体行动",
                "用结果或复盘收尾",
            ],
        )

    star_markers = {
        "背景": ("背景", "当时", "项目", "情况"),
        "任务": ("目标", "任务", "需要", "负责"),
        "行动": ("我做", "我负责", "采取", "首先", "随后", "推动"),
        "结果": ("结果", "最终", "提升", "降低", "完成", "%"),
    }
    present = [name for name, markers in star_markers.items() if any(marker in value for marker in markers)]
    structure = min(100, 35 + len(present) * 16)
    relevant_terms = [*job.must_skills, *job.preferred_skills, *job.keywords[:8]]
    relevance_hits = sum(compact(term) in compact(value) for term in relevant_terms)
    relevance = min(100, 45 + relevance_hits * 10) if relevant_terms else 70
    evidence_signals = len(re.findall(r"\d+(?:\.\d+)?%?|\d+人|\d+天|\d+周|\d+个月", value))
    evidence = min(100, 35 + evidence_signals * 18)
    sentence_count = len([item for item in re.split(r"[。！？!?\n]", value) if item.strip()])
    clarity = 90 if 4 <= sentence_count <= 10 else 72 if 2 <= sentence_count <= 14 else 55
    total = round(structure * 0.3 + relevance * 0.3 + evidence * 0.25 + clarity * 0.15)

    strengths = []
    if len(present) >= 3:
        strengths.append("回答具备较完整的 STAR 结构")
    if relevance_hits:
        strengths.append("能主动关联岗位要求")
    if evidence_signals:
        strengths.append("使用了数字或范围说明成果")
    improvements = []
    missing_star = [name for name in star_markers if name not in present]
    if missing_star:
        improvements.append(f"补充 STAR 要素：{'、'.join(missing_star)}")
    if not evidence_signals:
        improvements.append("补充一个可验证结果：规模、效率、转化、准确率或反馈")
    if not relevance_hits:
        improvements.append("结尾明确连接岗位的一项核心要求")
    if sentence_count > 14:
        improvements.append("压缩背景，把更多篇幅留给你的具体行动和结果")

    return InterviewScore(
        total=total,
        structure=structure,
        relevance=relevance,
        evidence=evidence,
        clarity=clarity,
        strengths=strengths or ["回答已经提供了可继续打磨的真实素材"],
        improvements=improvements or ["进一步压缩到 90—120 秒，并把结果放在结尾强调"],
        improved_answer_outline=[
            "结论：我具备什么能力",
            "背景与任务：为什么重要",
            "行动：我具体做了什么",
            "结果与复盘：数据、影响、迁移到本岗位",
        ],
    )


def top_skill_priorities(
    analyses: list[tuple[JobRequirements, MatchAnalysis]], limit: int = 3
) -> list[dict[str, Any]]:
    missing_counts: Counter[str] = Counter()
    preferred_counts: Counter[str] = Counter()
    role_examples: dict[str, list[str]] = {}
    for job, analysis in analyses:
        for skill in analysis.missing_skills:
            missing_counts[skill] += 2
            role_examples.setdefault(skill, []).append(job.title)
        for skill in analysis.preferred_gaps:
            preferred_counts[skill] += 1
            role_examples.setdefault(skill, []).append(job.title)
    total = missing_counts + preferred_counts
    return [
        {
            "skill": skill,
            "priority": score,
            "roles": unique(role_examples.get(skill, []))[:3],
            "plan": learning_plan(skill),
        }
        for skill, score in total.most_common(limit)
    ]


def learning_plan(skill: str) -> list[str]:
    if skill in LEARNING_PATHS:
        return LEARNING_PATHS[skill]
    return [
        f"第1周：完成{skill}的基础课程或官方教程，整理一页知识地图。",
        f"第2周：做一个与目标岗位相关的{skill}小项目，保留过程记录。",
        "第3周：把项目改写成 STAR 经历，补充数据、截图或可访问成果。",
        "第4周：完成两轮模拟面试，并把反馈更新到简历与答案库。",
    ]


def role_reference(job: JobRequirements) -> dict[str, Any] | None:
    """返回岗位画像与薪资参考，供界面展示。"""
    playbook = find_role_playbook(f"{job.title} {' '.join(job.responsibilities)}")
    if not playbook:
        return None
    return {
        "role": playbook,
        "salary": salary_for_role(f"{job.title} {' '.join(job.responsibilities)}"),
    }


def generate_cover_letter(profile: ResumeProfile, job: JobRequirements, matched: list[str], missing: list[str]) -> str:
    """生成一封结构完整的求职信草稿（中文）。"""
    identity = (
        "、".join(value for value in [profile.school, profile.major, profile.education] if value)
        or "相关专业背景"
    )
    ability = "、".join(matched[:3]) or "快速学习与沟通协作"
    proof = profile.highlights[0] if profile.highlights else "在课程、项目或实习中完成了从分析到交付的完整闭环"
    gap_line = (
        f"针对岗位目前我仍在补强的{('、'.join(missing[:3]))}，我已经制定并开始执行对应的学习与项目计划。"
        if missing
        else "我对岗位核心任务所需的技能已有对应积累。"
    )
    return (
        f"尊敬的招聘负责人：\n\n"
        f"您好！我是{profile.name or '候选人'}，{identity}。我看到贵司正在招聘「{job.title}」，"
        f"在仔细研读岗位职责后，认为自己的经历与能力与该岗位较为匹配，特此申请。\n\n"
        f"【我能带来的价值】我最相关的核心能力是{ability}。{proof}。"
        f"这与岗位要求的“{job.responsibilities[0] if job.responsibilities else '核心职责'}”高度对应，"
        f"我相信能够较快上手并为团队产出可衡量的结果。\n\n"
        f"【为什么是贵司】我对贵司所在的业务方向持续关注，认同用结果说话、注重协作与成长的文化，"
        f"希望能在这样的环境中把已有经验转化为实际贡献。\n\n"
        f"【关于差距与计划】{gap_line}\n\n"
        f"期待有机会进一步沟通，也欢迎随时安排面试。感谢您的时间！\n\n"
        f"此致\n敬礼！\n\n{profile.name or '候选人'}"
    )


def rewrite_bullets(profile: ResumeProfile, job: JobRequirements, matched: list[str]) -> list[str]:
    """把候选人的成果改写成 STAR 结构的简历要点草稿。"""
    bullets: list[str] = []
    role = job.title or "目标岗位"
    for highlight in profile.highlights[:3]:
        bullets.append(
            f"【动作】围绕{role}的核心任务，主导/参与{highlight}；"
            f"【方法】结合{'、'.join(matched[:2]) or '相关工具与方法'}；"
            f"【结果】给出可验证的数据或交付物（补充具体数字、覆盖范围或周期）。"
        )
    if not profile.highlights:
        bullets.append(
            f"【动作】针对{role}的核心职责，完成一个端到端项目；"
            f"【方法】使用{'、'.join(matched[:3]) or '相关技能'}；"
            f"【结果】量化成果（如效率、准确率、覆盖用户数、交付周期）。"
        )
    bullets.append(
        "通用模板：在 [时间/场景] 中，我通过 [具体动作 + 工具/方法]，"
        "实现了 [可验证结果]，并沉淀了 [方法/复盘]，可迁移到本岗位的 [对应职责]。"
    )
    return bullets[:5]


def generate_model_answer(question: str, profile: ResumeProfile, job: JobRequirements) -> str:
    """为一道面试题生成 STAR 示范回答骨架（需要候选人填入真实经历）。"""
    evidence = profile.highlights[0] if profile.highlights else "（填入你真实的项目/实习成果，附上数字）"
    return (
        f"【结论】一句话直接回答：我具备/会这样做，因为我曾{evidence}。\n"
        f"【背景】当时的目标是（交代时间、团队、要解决的问题）。\n"
        f"【行动】我具体做了三件事：1）…；2）…；3）…（强调“我”的动作，而非团队泛述）。\n"
        f"【结果】最终带来（量化结果：效率/准确率/覆盖/成本）的变化，并从中学到（复盘）。\n"
        f"【迁移】这段经历与{job.title}的“{job.responsibilities[0] if job.responsibilities else '核心职责'}”"
        f"直接相关，因此我有信心快速上手。"
    )


def jd_health_check(job_text: str) -> dict[str, list[str]]:
    """对 JD 做风险体检，返回命中的风险项与加分项。"""
    value = clean_text(job_text)
    red: list[str] = []
    keyword_map = {
        "付费陷阱": ("培训费", "押金", "服装费", "保证金", "先交钱", "付费"),
        "外包/培训贷": ("培训后上岗", "贷款培训", "包就业", "培训贷", "岗前培训收费"),
        "薪资倒挂": ("面议", "薪资面议", "待遇从优", "工资面议"),
        "信息缺失": (),
    }
    for title, _desc in JD_RED_FLAGS:
        for keyword in keyword_map.get(title, ()):
            if keyword in value:
                red.append(f"{title}：{dict(JD_RED_FLAGS)[title]}")
                break
    # 通用长度与结构判断
    if len(value) < 120:
        red.append("信息过简：JD 过短，关键信息（职责/要求/薪资）缺失。")
    if "薪资" not in value and "待遇" not in value and "工资" not in value and "薪酬" not in value:
        red.append("薪资未明确：建议投递前主动确认薪酬范围。")
    red = unique(red)
    return {"red": red, "green": list(JD_GREEN_FLAGS)}


def analysis_to_json(profile: ResumeProfile, job: JobRequirements, analysis: MatchAnalysis) -> str:
    payload = {"profile": asdict(profile), "job": asdict(job), "analysis": asdict(analysis)}
    payload["profile"]["raw_text"] = "（为保护隐私，导出报告不包含完整简历原文）"
    payload["job"]["source_text"] = job.source_text[:10_000]
    return json.dumps(payload, ensure_ascii=False, indent=2)
