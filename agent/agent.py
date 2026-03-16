"""
VC Scout – AI Agent
Implements VCScoutAgent: gathers market signals via tools and uses an LLM
to rank startup opportunities, detect sector trends, and generate reports.
"""

import json
import re
from typing import Any

from agent.tools import get_trending_repos, get_readme, get_reddit_posts, get_news


# ──────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────

ANALYST_PROMPT = """\
You are an expert venture capitalist and startup analyst.

Market signals gathered for topic: "{topic}"

━━━ GITHUB REPOSITORIES ━━━
{github}

━━━ REDDIT DISCUSSIONS ━━━
{reddit}

━━━ STARTUP NEWS ━━━
{news}

Identify and rank the top startup opportunities from these signals.

Return ONLY valid JSON — no markdown fences, no extra text:

{{
  "startups": [
    {{
      "startup_name": "",
      "startup_idea": "",
      "market_potential": 0,
      "reason": "",
      "investment_score": 0,
      "unicorn_probability": 0,
      "risk_level": "",
      "key_signals": []
    }}
  ]
}}

Rules:
- market_potential, investment_score, unicorn_probability → integers 1–10
- risk_level → "low" | "medium" | "high"
- key_signals → list of 2–3 short strings citing actual signals (repo names, post titles, headlines)
- Rank from highest to lowest investment_score
"""

SECTOR_PROMPT = """\
You are a market trend analyst.

Given the following signals on the topic "{topic}", identify the top emerging sectors.

GitHub repos:
{github}

Reddit posts:
{reddit}

News headlines:
{news}

Return ONLY valid JSON — no markdown, no prose:

{{
  "sectors": [
    {{
      "name": "",
      "signal_count": 0,
      "trend": "hot" | "rising" | "emerging",
      "summary": ""
    }}
  ]
}}

Rules:
- Identify 3–5 distinct sectors
- signal_count = total relevant signals (repos + posts + articles) for that sector
- trend: "hot" (dominant), "rising" (growing fast), "emerging" (early stage)
- summary: one sentence explaining why this sector is notable
"""

REPORT_PROMPT = """\
You are a senior venture capital analyst writing an investment memo.

Topic: {topic}

Startup Opportunities (ranked):
{startups}

Market Signals:
- GitHub: {github_count} trending repositories
- Reddit: {reddit_count} community discussions
- News: {news_count} articles

Write a professional plain-text investment report with these sections:
1. Executive Summary
2. Market Opportunity
3. Top Startup Opportunities (ranked, with scores)
4. Signal Analysis
5. Risk Assessment
6. Investment Recommendation

Use plain text only — no markdown, no special characters. Be specific and data-driven.
"""


# ──────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────

class VCScoutAgent:
    """
    A lightweight AI agent for venture capital signal analysis.

    Public methods
    ──────────────
    analyze(topic)          → { topic, signals, analysis }
    detect_sectors(signals) → { sectors: [...] }
    generate_report(...)    → plain-text investment memo (str)
    gather_signals(topic)   → raw signals dict
    """

    DEFAULT_TOOLS: list[str] = ["get_trending_repos", "get_reddit_posts", "get_news"]

    def __init__(self, llm_client, model: str = "llama-3.1-8b-instant"):
        self.client = llm_client
        self.model  = model

        self.tool_registry: dict[str, Any] = {
            "get_trending_repos": get_trending_repos,
            "get_reddit_posts":   get_reddit_posts,
            "get_news":           get_news,
            "get_readme":         get_readme,
        }

    # ── Shared helpers ─────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown fences and extract the first JSON object."""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        match   = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            return json.loads(match.group())
        return {"raw_output": cleaned}

    # ── Step 1: Gather signals ─────────────────────────────────────

    def gather_signals(self, topic: str) -> dict[str, list]:
        """
        Invoke each primary tool and return { tool_name: results }.
        Individual tool errors are caught so the agent keeps running.
        """
        signals: dict[str, list] = {}

        for tool_name in self.DEFAULT_TOOLS:
            tool_fn = self.tool_registry.get(tool_name)
            if tool_fn is None:
                continue
            try:
                result = tool_fn(topic)

                if tool_name == "get_trending_repos":
                    readme_fn = self.tool_registry.get("get_readme")
                    for repo in result:
                        try:
                            repo["readme"] = readme_fn(repo["owner"], repo["name"]) if readme_fn else ""
                        except Exception:
                            repo["readme"] = "README unavailable"

                signals[tool_name] = result[:3]

            except Exception as exc:
                signals[tool_name] = [{"error": str(exc)}]

        return signals

    # ── Step 2: Startup analysis ───────────────────────────────────

    def analyze(self, topic: str) -> dict:
        """Gather signals → build prompt → call LLM → return ranked startups."""
        signals = self.gather_signals(topic)

        github = signals.get("get_trending_repos", [])
        reddit = signals.get("get_reddit_posts",   [])
        news   = signals.get("get_news",           [])

        prompt = ANALYST_PROMPT.format(
            topic=topic,
            github=json.dumps(github, indent=2),
            reddit=json.dumps(reddit, indent=2),
            news=json.dumps(news,     indent=2),
        )

        raw      = self._call_llm(prompt)
        analysis = self._parse_json(raw)

        return {
            "topic":    topic,
            "signals":  signals,
            "analysis": analysis,
        }

    # ── Step 3: Sector trend detection ────────────────────────────

    def detect_sectors(self, topic: str, signals: dict) -> dict:
        """
        Given already-gathered signals, ask the LLM to identify
        the top emerging sectors with trend labels.
        Returns { sectors: [{ name, signal_count, trend, summary }] }
        """
        github = signals.get("get_trending_repos", [])
        reddit = signals.get("get_reddit_posts",   [])
        news   = signals.get("get_news",           [])

        prompt = SECTOR_PROMPT.format(
            topic=topic,
            github=json.dumps(github, indent=2),
            reddit=json.dumps(reddit, indent=2),
            news=json.dumps(news,     indent=2),
        )

        raw = self._call_llm(prompt)
        return self._parse_json(raw)

    # ── Step 4: Investment report ──────────────────────────────────

    def generate_report(self, topic: str, signals: dict, analysis: dict) -> str:
        """
        Generate a professional plain-text VC investment memo.
        Returns a string (not JSON).
        """
        startups = analysis.get("startups", [])

        prompt = REPORT_PROMPT.format(
            topic=topic,
            startups=json.dumps(startups, indent=2),
            github_count=len(signals.get("get_trending_repos", [])),
            reddit_count=len(signals.get("get_reddit_posts",   [])),
            news_count=len(signals.get("get_news",             [])),
        )

        return self._call_llm(prompt)
