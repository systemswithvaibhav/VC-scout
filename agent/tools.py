"""
VC Scout – Tool definitions
Each function is a standalone tool the agent can invoke.
"""

import os
import base64
import requests


# ──────────────────────────────────────────────
# GitHub tools
# ──────────────────────────────────────────────

def get_trending_repos(topic: str) -> list[dict]:
    """
    Fetch the top 3 GitHub repositories for a topic,
    sorted by star count.
    """
    url = (
        f"https://api.github.com/search/repositories"
        f"?q={topic}&sort=stars&order=desc&per_page=3"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    repos = []
    for repo in data.get("items", []):
        repos.append({
            "name":        repo["name"],
            "owner":       repo["owner"]["login"],
            "stars":       repo["stargazers_count"],
            "forks":       repo["forks_count"],
            "issues":      repo["open_issues_count"],
            "language":    repo["language"],
            "updated":     repo["updated_at"],
            "description": repo["description"],
            "url":         repo["html_url"],
        })
    return repos


def get_readme(owner: str, repo: str) -> str:
    """
    Fetch and decode the README for a GitHub repo.
    Returns the first 500 characters (to keep the LLM prompt compact).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return "No README available"

    content = response.json().get("content", "")
    decoded = base64.b64decode(content).decode("utf-8")
    return decoded[:500]


# ──────────────────────────────────────────────
# Reddit tool
# ──────────────────────────────────────────────

def get_reddit_posts(topic: str) -> list[dict]:
    """
    Fetch the top 5 Reddit posts related to a topic and startups.
    """
    url = f"https://www.reddit.com/search.json?q={topic} startup&limit=5"
    headers = {"User-Agent": "vc-scout/1.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    posts = []
    for post in data["data"]["children"]:
        p = post["data"]
        posts.append({
            "title":    p["title"],
            "upvotes":  p["ups"],
            "comments": p["num_comments"],
        })
    return posts


# ──────────────────────────────────────────────
# News tool
# ──────────────────────────────────────────────

def get_news(topic: str) -> list[dict]:
    """
    Fetch the top 5 startup-related news articles via NewsAPI.
    """
    news_key = os.getenv("NEWS_API_KEY")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={topic} startup&apiKey={news_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    articles = []
    for article in data.get("articles", [])[:5]:
        articles.append({
            "title":  article["title"],
            "source": article["source"]["name"],
        })
    return articles
