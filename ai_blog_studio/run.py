import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
ROOT_DIR = BACKEND_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env
try:
    from dotenv import load_dotenv

    env_path = BACKEND_DIR.parent / ".env"
    load_dotenv(dotenv_path=env_path)

except ImportError:
    pass

import os
import sentry_sdk

sentry_dsn = os.getenv("SENTRY_DSN")

if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
    )

from ai_blog_studio.graph.graph import graph


def today_ist() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(
        description="AI Blog Studio LangGraph Article Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Generate today's article:
  python backend/run.py

  # Generate for a specific date:
  python backend/run.py --date 2026-03-07

  # Dry run:
  python backend/run.py --dry-run

  # Generate AI News article:
  python backend/run.py --ainews
        """,
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (default: today in IST)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode: skip Groq calls and file writes",
    )

    parser.add_argument(
        "--ainews",
        action="store_true",
        help="Run the AI News gathering and generation graph",
    )

    args = parser.parse_args()

    date_str = args.date or today_ist()
    dry_run = args.dry_run
    run_ainews = args.ainews

    print(f"\n{'=' * 55}")
    print("  AI Blog Studio — LangGraph Article Generator")
    print(f"  Date    : {date_str}")
    print(f"  Dry run : {dry_run}")
    print(f"{'=' * 55}")

    initial_state = {
        "date": date_str,
        "dry_run": dry_run,
    }

    if run_ainews:
        initial_state["domain"] = "ainews"

    config = {
        "configurable": {
            "thread_id": "ai_blog_studio-1"
        }
    }

    final_state = graph.invoke(
        initial_state,
        config=config
    )

    print(f"\n{'=' * 55}")

    if dry_run:

        print(
            "  [DRY RUN] Pipeline completed — "
            "no files were written."
        )

        domain = final_state.get("domain", "?")
        topic = final_state.get("topic", "?")
        slug = final_state.get("slug", "generated-article")

        print(f"  Chosen Domain : {domain}")
        print(f"  Chosen Topic  : {topic}")
        print("  Would have generated:")
        print(f"    -> data/blogs/{domain}/{slug}.md")
        print(f"    -> data/blogs/{domain}/articles.json")

    else:

        domain = final_state.get("domain", "?")
        title = final_state.get("title")
        md_path = final_state.get("md_path")
        read_time = final_state.get("read_time", "?")

        generation_failed = final_state.get(
            "generation_failed",
            False
        )

        revision_needed = final_state.get(
            "revision_needed",
            False
        )

        # Successful approval + successful save
        if (
            not generation_failed
            and not revision_needed
            and title
            and md_path
        ):

            print("  🎉 Done! Article generated successfully.")
            print(f"  Title     : {title}")
            print(f"  Domain    : {domain}")
            print(f"  Read time : {read_time}")
            print(f"  File      : {md_path}")

        else:

            print("  ❌ Article was NOT generated successfully.")
            print(f"  Domain    : {domain}")
            print(
                f"  Revisions : "
                f"{final_state.get('revision_count', 0)}"
            )

            feedback = final_state.get("validator_feedback")

            if feedback:
                print("\n  Validator feedback:")
                print(f"  {feedback}")

            print("\n  No article was published.")

    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
