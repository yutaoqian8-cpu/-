from __future__ import annotations

import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_VERSION = "1.0"
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
        "如果入职后两周内要交付第一个成果，你会如何拆解和安排？",
    ]
    questions.extend(f"岗位强调{skill}，请说明你的理解，并给出一个真实使用场景。" for skill in gaps[:3])
    if job.responsibilities:
        questions.append(f"针对“{job.responsibilities[0]}”，你会如何制定第一周的行动计划？")
    return unique(questions)[:8]


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
        evidence.append(f"学历要求为{job.education}及对应口径，当前填写为{profile.education or '未填写'}")
    if job.min_years:
        evidence.append(f"岗位要求约{job.min_years:g}年经验，当前填写为{profile.experience_years:g}年")
    if not job.must_skills:
        evidence.append("岗位描述中没有识别到明确技能词，技能项按保守分值处理")

    next_actions = []
    if missing:
        next_actions.append(f"先用一个小项目验证并补齐：{'、'.join(missing[:3])}")
    next_actions.extend(
        [
            "按岗位关键词重排简历，只保留可验证的能力证据",
            "准备 3 个 STAR 故事：成果、协作、失败复盘",
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
    return [
        f"第1周：完成{skill}基础课程或官方教程，并整理一页知识地图。",
        f"第2周：做一个与目标岗位相关的{skill}小项目，保留过程记录。",
        "第3周：把项目改写成 STAR 经历，补充数据、截图或可访问成果。",
        "第4周：完成两轮模拟面试，并将反馈更新到简历与答案库。",
    ]


def analysis_to_json(profile: ResumeProfile, job: JobRequirements, analysis: MatchAnalysis) -> str:
    payload = {"profile": asdict(profile), "job": asdict(job), "analysis": asdict(analysis)}
    payload["profile"]["raw_text"] = "（为保护隐私，导出报告不包含完整简历原文）"
    payload["job"]["source_text"] = job.source_text[:10_000]
    return json.dumps(payload, ensure_ascii=False, indent=2)
