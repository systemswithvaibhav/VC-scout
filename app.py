from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import os

from agent.tools import get_trending_repos, get_readme
from agent.agent import VCScoutAgent


# ── Bootstrap ──────────────────────────────────────────────────────
load_dotenv()

app        = Flask(__name__)
CORS(app)

llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
agent      = VCScoutAgent(llm_client)


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    return "VC Scout AI – Agent Mode 🚀"


@app.route("/trending")
def trending():
    """Raw GitHub trending repos + README snippets for a topic."""
    topic = request.args.get("topic", "startup")
    repos = get_trending_repos(topic)
    for repo in repos:
        repo["readme"] = get_readme(repo["owner"], repo["name"])
    return jsonify(repos)


@app.route("/analyze")
def analyze():
    """
    Full agent analysis — gather signals + rank startup opportunities.

    Response: { topic, signals, analysis: { startups: [...] } }
    """
    try:
        topic  = request.args.get("topic", "startup")
        result = agent.analyze(topic)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/signals")
def signals():
    """
    Return only the raw gathered signals for a topic.
    Useful for the Signal Breakdown panel without triggering LLM analysis.

    Response: { topic, signals: { get_trending_repos, get_reddit_posts, get_news } }
    """
    try:
        topic = request.args.get("topic", "startup")
        raw   = agent.gather_signals(topic)
        return jsonify({"topic": topic, "signals": raw})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sectors")
def sectors():
    """
    Detect trending sectors from signals for a topic.

    Response: { topic, sectors: [{ name, signal_count, trend, summary }] }
    """
    try:
        topic   = request.args.get("topic", "startup")
        raw     = agent.gather_signals(topic)
        result  = agent.detect_sectors(topic, raw)
        return jsonify({"topic": topic, "sectors": result.get("sectors", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report")
def report():
    """
    Generate a plain-text investment report for a topic.
    Triggers full pipeline: gather_signals → analyze → generate_report.

    Response: text/plain — downloadable VC memo
    """
    try:
        topic    = request.args.get("topic", "startup")
        # Run full analysis first to get both signals and startup rankings
        result   = agent.analyze(topic)
        report_text = agent.generate_report(
            topic=topic,
            signals=result["signals"],
            analysis=result["analysis"],
        )
        return Response(
            report_text,
            mimetype="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="vc-report-{topic}.txt"'
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)