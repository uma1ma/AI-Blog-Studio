from typing import TypedDict, List, Dict, Any


class BlogState(TypedDict, total=False):

    # Core metadata
    domain: str
    topic: str
    subtopics: str
    date: str
    schedule: Dict[str, Any]
    dry_run: bool

    # Research context
    recent_blogs: List[str]
    news_data: str

    # Generated content
    title: str
    description: str
    tags: List[str]
    slug: str
    content: str
    read_time: str

    # Validation
    validator_feedback: str
    revision_count: int
    revision_needed: bool

    # Workflow observability
    agent_trace: List[str]

    # Final output
    md_path: str
    skipped: bool
