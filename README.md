# AI Blog Studio

**AI Blog Studio** is a multi-agent AI system for generating, validating, revising, and publishing technical blog articles.

The project uses **LangGraph** to coordinate specialized agents and **Groq-hosted LLMs** to generate technical content. Generated articles are passed through an editorial validation step, and rejected drafts are automatically sent back to the appropriate generation agent for revision.

## Features

* **Multi-agent workflow** built with LangGraph
* **Automatic topic selection** based on the selected content domain and recent article history
* **AI article generation** using Groq-hosted LLMs
* **Editorial validation** of generated articles
* **Automatic revision loop** when an article does not meet the publication criteria
* **Structured metadata generation** including title, description, slug, and reading time
* **Markdown article publishing**
* **Local filesystem storage** for generated articles and article metadata
* **Domain-based organization** of generated content
* **Static frontend** for browsing generated articles
* **Dry-run mode** for testing the workflow without making an LLM generation request

## How It Works

The generation pipeline follows a simple agent workflow:

```text
                    ┌─────────────────┐
                    │   Start / Run   │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Topic Selection   │
                  │  / Existing Topic  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Tutorial / News   │
                  │       Agent         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Article Draft     │
                  │     Generation      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Validator Agent    │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                  Reject            Approve
                    │                 │
                    ▼                 ▼
              Regenerate         Save Article
                    │                 │
                    └───────►─────────┘
```

### Agents

#### Tutorial Agent

Responsible for:

* Selecting a suitable technical topic when one is not provided
* Generating the complete article
* Incorporating validator feedback during revisions
* Calculating estimated reading time

#### News Agent

Handles articles for the AI/news domain using the same LangGraph workflow.

#### Validator Agent

Acts as the final editorial quality checker.

It evaluates whether the generated article:

* Is substantial enough to publish
* Clearly addresses the requested topic
* Has a coherent structure
* Uses sensible Markdown
* Has a complete ending
* Avoids obvious placeholders or severe formatting issues

If the article is rejected, the validator provides feedback and the generation agent produces a revised version.

## Tech Stack

* **Python**
* **LangGraph** — agent workflow orchestration
* **LangChain**
* **Groq** — LLM inference
* **Pydantic** — configuration and validation
* **uv** — Python dependency management
* **Markdown** — generated article format
* **JavaScript / HTML / CSS** — static frontend

## Project Structure

```text
AI-Blog-Studio/
│
├── ai_blog_studio/
│   ├── agents/
│   │   ├── tutorial_agent/
│   │   ├── news_agent/
│   │   └── validator_agent/
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── graph/
│   │   ├── graph.py
│   │   └── state.py
│   │
│   ├── services/
│   │   ├── llm.py
│   │   ├── storage.py
│   │   └── prompt_manager.py
│   │
│   ├── web/
│   │   ├── js/
│   │   └── ...
│   │
│   └── run.py
│
├── data/
│   └── blogs/
│       └── <domain>/
│           ├── articles.json
│           └── *.md
│
├── .env.example
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## Requirements

* Python 3.13+
* `uv`
* A Groq API key

## Installation

Clone the repository:

```bash
git clone https://github.com/uma1ma/AI-Blog-Studio.git
cd AI-Blog-Studio
```

Install the project dependencies:

```bash
uv sync
```

Create your environment file:

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key.

For example:

```text
llm__api_key="your_groq_api_key"
```

Do **not** commit your `.env` file or expose your API key publicly.

## Running the Project

Run the complete article-generation pipeline with:

```bash
uv run python -m ai_blog_studio.run
```

The system will:

1. Select a content domain and topic when required.
2. Generate an article using the appropriate agent.
3. Send the draft to the validator.
4. Reject and regenerate the article if a major issue is detected.
5. Approve the article when it meets the publication criteria.
6. Save the final article as Markdown.
7. Update the domain's `articles.json` metadata.

### Dry Run

To test the workflow without performing the full LLM generation:

```bash
uv run python -m ai_blog_studio.run --dry-run
```

## Generated Content

Generated articles are stored locally under:

```text
data/blogs/
```

For example:

```text
data/blogs/ml/
├── articles.json
└── diffusion-models-generative-ai-theory-training.md
```

The metadata file contains information such as:

* Article title
* Topic
* Subtopics
* Description
* URL slug
* Date
* Reading time
* Markdown file location

## Example Workflow

A successful generation can include multiple validation cycles.

For example:

```text
TutorialAgent
      ↓
Generate article
      ↓
ValidatorAgent
      ↓
Rejected
      ↓
Validator feedback
      ↓
TutorialAgent
      ↓
Generate revised article
      ↓
ValidatorAgent
      ↓
Approved
      ↓
Save Markdown + metadata
```

This allows the system to improve an article automatically rather than simply accepting the first generated draft.

## Frontend

The project includes a lightweight static frontend for browsing generated articles.

Article data is loaded from the local `data/` directory, allowing the generated Markdown content and metadata to be viewed without requiring a cloud object-storage service.

## Configuration

The main configuration is handled through environment variables.

The `.env.example` file contains placeholders for the supported services.

The core LLM configuration uses:

```text
llm__api_key="your_groq_api_key"
```

Optional integrations such as content APIs, Opik tracing, and Sentry can be configured when required.

## Project Status

The core pipeline has been tested end-to-end:

* Topic selection: working
* Article generation: working
* Validator: working
* Revision workflow: working
* Markdown publishing: working
* Local article storage: working
* Article metadata generation: working
* Static frontend data loading: implemented

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

This repository is an adaptation of an MIT-licensed multi-agent blog-generation project. The original license and applicable attribution have been retained in accordance with the license terms.
