import json
import re
from datetime import datetime

from ai_blog_studio.graph.state import BlogState
from ai_blog_studio.services.llm import LLMAgentService
from ai_blog_studio.services.storage import LocalStorageService
from ai_blog_studio.services.prompt_manager import prompt_manager
from .prompts import VALIDATOR_PROMPT


def _extract_json(text: str) -> dict:
    raw = text.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])

    raise json.JSONDecodeError(
        "No valid JSON object found",
        raw,
        0
    )


def _safe_slug(value: str) -> str:
    slug = value.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def _build_validation_sample(content: str) -> str:
    """
    Keep validator requests comfortably below Groq's TPM limit.
    The validator receives the beginning and end of the article plus
    structural information rather than the entire article.
    """
    lines = content.splitlines()

    first = content[:5000]
    last = content[-3000:] if len(content) > 8000 else ""

    headings = [
        line.strip()
        for line in lines
        if line.strip().startswith("#")
    ]

    code_blocks = content.count("```")
    table_lines = sum(
        1 for line in lines
        if "|" in line
    )

    structure = (
        "\n\nARTICLE STRUCTURE SUMMARY:\n"
        f"- Word count: {len(content.split())}\n"
        f"- Headings found: {len(headings)}\n"
        f"- Code fence markers: {code_blocks}\n"
        f"- Table lines: {table_lines}\n"
        f"- Headings: {', '.join(headings[:20])}\n"
    )

    if last:
        return (
            "BEGINNING OF ARTICLE:\n"
            + first
            + structure
            + "\nEND OF ARTICLE:\n"
            + last
        )

    return first + structure


def validator_node(state: BlogState) -> BlogState:
    print("  => [ValidatorAgent] Running...")

    trace = list(state.get("agent_trace", []))
    trace.append("Validator Agent")

    if state.get("dry_run"):
        print("  [DRY RUN] Simulating Approval and Metadata Gen.")

        topic = state.get("topic", "AI Blog Article")
        title = topic[:70]
        description = f"A practical guide to {topic}."[:160]
        slug = _safe_slug(topic)

        return {
            **state,
            "approved": True,
            "revision_needed": False,
            "validator_feedback": "",
            "title": title,
            "description": description,
            "slug": slug,
            "agent_trace": trace,
        }

    content = state.get("content", "")
    topic = state.get("topic", "AI Blog Article")
    current_revision = state.get("revision_count", 0)

    validation_content = _build_validation_sample(content)

    validator_prompt = prompt_manager.get_prompt(
        prompt_name="Validator_Prompt",
        fallback_prompt=VALIDATOR_PROMPT,
        topic=topic,
        content=validation_content,
    )

    llm_service = LLMAgentService(temperature=0.1)

    try:
        response = llm_service.llm.invoke(validator_prompt)
        result = _extract_json(response.content)

        approved = bool(result.get("approved", False))
        feedback = str(result.get("feedback", "") or "")
        title = str(result.get("title", "") or "")
        description = str(result.get("description", "") or "")
        slug = str(result.get("slug", "") or "")

    except Exception as e:
        print(f"  [WARN] Validator failed to return valid JSON: {e}")
        print("  [WARN] Treating draft as rejected.")

        return {
            **state,
            "revision_needed": True,
            "validator_feedback": (
                "Validator temporarily failed to process the draft. "
                "Please regenerate the article."
            ),
            "revision_count": current_revision + 1,
            "agent_trace": trace,
        }

    if not approved:
        print("  [AGENT] Draft REJECTED.")
        print(f"  [AGENT] Feedback: {feedback}")

        return {
            **state,
            "approved": False,
            "revision_needed": True,
            "validator_feedback": feedback or "Please revise the article.",
            "revision_count": current_revision + 1,
            "agent_trace": trace,
        }

    # Generate metadata after approval.
    if not title:
        title = topic[:70]

    if not description:
        description = f"A practical guide to {topic}."[:160]

    if not slug:
        slug = _safe_slug(title)

    title = title[:70].strip()
    description = description[:160].strip()
    slug = _safe_slug(slug)

    domain = state.get("domain", "ml")
    storage = LocalStorageService()

    date_str = state.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    article = {
        "title": title,
        "topic": topic,
        "subtopics": state.get("subtopics", ""),
        "description": description,
        "slug": slug,
        "date": date_str,
        "read_time": state.get("read_time", "1 min"),
        "file": f"blogs/{domain}/{slug}.md",
    }

    md_key = f"blogs/{domain}/{slug}.md"

    saved_md = storage.put_object(
        md_key,
        content,
        content_type="text/markdown",
    )

    if not saved_md:
        print("  [ERROR] Failed to save article Markdown.")

        return {
            **state,
            "generation_failed": True,
            "revision_needed": False,
            "validator_feedback": "Article was approved but could not be saved.",
            "agent_trace": trace,
        }

    articles = storage.get_articles_json(domain)

    articles = [
        item for item in articles
        if item.get("slug") != slug
    ]

    articles.append(article)

    saved_json = storage.save_articles_json(domain, articles)

    if not saved_json:
        print("  [ERROR] Article saved but articles.json could not be updated.")

        return {
            **state,
            "generation_failed": True,
            "revision_needed": False,
            "validator_feedback": "Article saved but metadata could not be updated.",
            "agent_trace": trace,
        }

    print("  [AGENT] Draft APPROVED.")
    print(f"  [AGENT] Title: {title}")
    print(f"  [AGENT] Saved: data/{md_key}")

    return {
        **state,
        "approved": True,
        "revision_needed": False,
        "generation_failed": False,
        "validator_feedback": "",
        "title": title,
        "description": description,
        "slug": slug,
        "md_path": f"data/{md_key}",
        "agent_trace": trace,
    }
