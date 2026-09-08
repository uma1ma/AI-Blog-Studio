import json
from pathlib import Path
from typing import Optional, List, Dict, Any


class LocalStorageService:
    """
    Local filesystem storage service.

    Keeps the same interface used by the agents, but stores
    articles and JSON data inside the local data/ directory.
    """

    def __init__(self):
        self.base_path = Path("data")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_object(self, key: str) -> Optional[str]:
        """Fetch raw string data from local storage."""
        path = self._path(key)

        if not path.exists():
            return None

        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[ERROR] Failed to read {key}: {e}")
            return None

    def put_object(
        self,
        key: str,
        data: str,
        content_type: str = "text/plain"
    ) -> bool:
        """Save string data to local storage."""
        path = self._path(key)

        try:
            path.write_text(data, encoding="utf-8")
            print(f"  ✅ Saved locally: {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch and parse JSON from local storage."""
        data = self.get_object(key)

        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to decode JSON from {key}. Starting fresh.")
                return []

        return []

    def get_articles_json(self, domain: str) -> List[Dict[str, Any]]:
        """Fetch the articles registry for a domain."""
        return self.get_json(f"blogs/{domain}/articles.json") or []

    def save_articles_json(
        self,
        domain: str,
        articles: List[Dict[str, Any]]
    ) -> bool:
        """Save the articles registry for a domain."""
        json_str = json.dumps(
            articles,
            indent=2,
            ensure_ascii=False
        )

        return self.put_object(
            f"blogs/{domain}/articles.json",
            json_str,
            content_type="application/json"
        )

    def get_recent_history(
        self,
        domain: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Fetch the N most recent articles for context."""
        articles = self.get_articles_json(domain)

        sorted_articles = sorted(
            articles,
            key=lambda x: x.get("date", ""),
            reverse=True
        )

        recent = sorted_articles[:limit]

        return [
            {
                "title": a.get("title"),
                "topic": a.get("topic"),
                "subtopics": a.get("subtopics", "")
            }
            for a in recent
        ]

    def get_all_domains_last_updated(self) -> Dict[str, str]:
        """Return the latest article date for every configured domain."""
        latest_dates = {}

        # These are the domains used by the application.
        domains = [
            "ml",
            "dl",
            "statistics",
            "nlp",
            "cv",
            "genai",
            "ainews",
        ]

        for domain_slug in domains:
            articles = self.get_articles_json(domain_slug)

            if not articles:
                latest_dates[domain_slug] = "Never"
            else:
                sorted_articles = sorted(
                    articles,
                    key=lambda x: x.get("date", ""),
                    reverse=True
                )

                latest_dates[domain_slug] = sorted_articles[0].get(
                    "date",
                    "Unknown"
                )

        return latest_dates


# Keep the original class name so the existing agents continue to work.
