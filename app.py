from flask import Flask, jsonify, request
import requests
from dotenv import load_dotenv
import os
import json
import re
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)

@app.route("/")
def home():
    return "VC Scout AI running"



# github repo fetching 
def get_trending_repos(topic):
    url = f"https://api.github.com/search/repositories?q={topic}&sort=stars&order=desc&per_page=3"
    response = requests.get(url)
    data = response.json()

    repos = []
    for repo in data["items"]:
       repos.append({
 "name": repo["name"],
 "owner": repo["owner"]["login"],   # add this
 "stars": repo["stargazers_count"],
 "forks": repo["forks_count"],
 "issues": repo["open_issues_count"],
 "language": repo["language"],
 "updated": repo["updated_at"],
 "description": repo["description"],
 "url": repo["html_url"]
})
        

    return repos

import base64
# github repo fetching 
def get_readme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    response = requests.get(url)

    if response.status_code != 200:
        return "No README available"

    data = response.json()

    content = data["content"]

    decoded = base64.b64decode(content).decode("utf-8")

    return decoded[:500]   # limit for AI



@app.route("/trending")
def trending():

    topic = request.args.get("topic", "startup")

    repos = get_trending_repos(topic)

    for repo in repos:
        readme = get_readme(repo["owner"], repo["name"])
        repo["readme"] = readme

    return jsonify(repos)


# reddit post fetching 
# import requests

def get_reddit_posts(topic):

    url = f"https://www.reddit.com/search.json?q={topic} startup&limit=5"
    headers = {
        "User-Agent": "vc-scout"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    posts = []

    for post in data["data"]["children"]:
        p = post["data"]

        posts.append({
            "title": p["title"],
            "upvotes": p["ups"],
            "comments": p["num_comments"]
        })

    return posts


# news api fetching
def get_news(topic):
    news_key = os.getenv("NEWS_API_KEY")


    url = f"https://newsapi.org/v2/everything?q={topic} startup&apiKey={news_key}"
    response = requests.get(url)

    data = response.json()

    articles = []

    for article in data["articles"][:5]:

        articles.append({
            "title": article["title"],
            "source": article["source"]["name"]
        })

    return articles 



@app.route("/analyze")
def analyze():

    try:
        topic = request.args.get("topic", "startup")

        github = get_trending_repos(topic)[:3]
        reddit = get_reddit_posts(topic)[:3]
        news = get_news(topic)[:3]

        prompt = f"""
You are a venture capitalist.

Rank startups from best to worst investment opportunity.

Analyze the following signals.

GitHub Repositories:
{github}

Reddit Discussions:
{reddit}

Startup News:
{news}

Return ONLY valid JSON.

{{
 "startups":[
  {{
   "startup_name":"",
   "startup_idea":"",
   "market_potential":"",
   "reason":"",
   "investment_score":"",
   "unicorn_probability":"",
   "risk_level":""
  }}
 ]
}}
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )

        analysis_text = response.choices[0].message.content

        analysis_text = analysis_text.replace("```json","").replace("```","").strip()

        match = re.search(r'\{[\s\S]*\}', analysis_text)

        if match:
            json_str = match.group()
            analysis = json.loads(json_str)
        else:
            analysis = {"raw_output": analysis_text}

        return jsonify({
            "topic": topic,
            "analysis": analysis
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)