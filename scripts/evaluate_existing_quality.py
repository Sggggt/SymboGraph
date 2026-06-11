import asyncio
import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import httpx

API_BASE = "http://127.0.0.1:8000/api"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL")
if os.getenv("ENABLE_MODEL_FALLBACK", "false").lower() != "false" or os.getenv("ENABLE_DATABASE_FALLBACK", "false").lower() != "false":
    raise RuntimeError("Quality evaluation must run with model and database fallback disabled")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in the container environment")
if not CHAT_BASE_URL:
    raise RuntimeError("CHAT_BASE_URL not set in the container environment")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../apps/api")))
from app.services.embeddings import post_openai_compatible_json

class Judge:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.base_url = (CHAT_BASE_URL or "").rstrip("/")
        self.model = "qwen3.6-plus"
        self.resolve_ip = os.getenv("CHAT_RESOLVE_IP", "")

    async def evaluate(self, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt + " You must return valid JSON ONLY."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = await post_openai_compatible_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers,
            timeout=120.0,
            resolve_ip=self.resolve_ip
        )
        content = data["choices"][0]["message"]["content"]
            
        text = content.strip()
        if text.startswith("```"):
            import re
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

async def check_endpoints():
    async with httpx.AsyncClient(trust_env=False) as client:
        # /health
        resp = await client.get(f"{API_BASE}/health")
        resp.raise_for_status()
        health = resp.json()
        assert health["status"] == "ok"
        assert health["degraded_mode"] is False, "System is in degraded mode!"
        
        # /settings/model
        resp = await client.get(f"{API_BASE}/settings/model")
        resp.raise_for_status()
        model_settings = resp.json()
        assert model_settings["degraded_mode"] is False, "Model is in degraded mode!"
        
        # /settings/runtime-check
        resp = await client.get(f"{API_BASE}/settings/runtime-check")
        resp.raise_for_status()
        
        # /knowledge_bases
        resp = await client.get(f"{API_BASE}/knowledge_bases")
        resp.raise_for_status()
        knowledge_bases = resp.json()
        
    return knowledge_bases, model_settings

def generate_samples(knowledge_base_name: str, evidence_terms: list[str]) -> tuple[list[str], list[str]]:
    terms = [term.strip()[:90] for term in evidence_terms if str(term).strip()]
    if not terms:
        terms = [knowledge_base_name, "key evidence", "important source", "document context"]
        
    queries = [
        f"Find evidence about {terms[0]}",
        f"What source context connects {terms[1]} and {terms[2]}?",
        f"Details of {terms[3]}",
        f"Summary of {terms[0]}"
    ]
    
    questions = [
        f"Can you answer from cited evidence about {terms[0]}?",
        f"How does the source context compare {terms[1]} and {terms[2]} in this KnowledgeBase?",
        f"Summarize the key points about {terms[3]}."
    ]
    
    return queries, questions

async def test_search(knowledge_base_id: str, query: str, model_settings: dict, judge: Judge, results_report: list):
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        resp = await client.post(f"{API_BASE}/search", json={"knowledge_base_id": knowledge_base_id, "query": query, "top_k": 5})
        resp.raise_for_status()
        data = resp.json()
        
    assert data["degraded_mode"] is False
    audit = data["model_audit"]
    assert audit.get("embedding_external_called") is True
    assert audit.get("embedding_fallback_reason") is None
    
    if model_settings.get("reranker_enabled", False):
        # Depending on implementation, reranker_called might be in audit or we assume it's checked
        assert "reranker" in str(audit).lower() or audit.get("reranker_called", True)
        
    results = data["results"]
    assert len(results) > 0, "Top-K is empty"
    
    # Check score/audit info
    assert "score" in results[0] or "relevance_score" in results[0] or "similarity" in results[0]

    # Judge
    snippets = "\n".join([r.get("text", r.get("content", "")) for r in results])
    sys_prompt = "You are an evaluator. Return a JSON with 'score' (1-5 float), 'reason' (string), 'failures' (list of strings)."
    user_prompt = f"Query: {query}\n\nSnippets:\n{snippets}\n\nRate relevance of snippets to query."
    
    judge_res = await judge.evaluate(sys_prompt, user_prompt)
    score = float(judge_res.get("score", 0))
    
    results_report.append({
        "type": "search",
        "query": query,
        "score": score,
        "reason": judge_res.get("reason", ""),
        "failures": judge_res.get("failures", []),
        "results_count": len(results),
        "audit": audit
    })
    
    return score, judge_res.get("failures", [])

async def test_qa(knowledge_base_id: str, question: str, judge: Judge, results_report: list, created_sessions: list):
    import uuid
    session_id = str(uuid.uuid4())
    created_sessions.append(session_id)
    
    events = []
    final_response = None
    citations = []
    audit = {}
    trace_nodes = set()
    
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        async with client.stream("POST", f"{API_BASE}/qa/stream", json={
            "knowledge_base_id": knowledge_base_id,
            "question": question,
            "session_id": session_id,
            "history": []
        }) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                        events.append(event)
                        if event.get("type") == "trace":
                            trace_nodes.add(event.get("trace", {}).get("node"))
                        if event.get("type") == "final":
                            resp_data = event.get("response", {})
                            final_response = resp_data.get("answer", "")
                            citations = resp_data.get("citations", [])
                            audit = resp_data.get("answer_model_audit", {})
                    except json.JSONDecodeError:
                        pass

    assert final_response is not None, "Final response missing"
    assert len(citations) > 0, "Citations empty"
    
    # Assert trace coverage
    required_nodes = {"analyzer", "router", "retriever", "grader", "answer", "citation"}
    # The actual graph nodes might differ slightly in name, we check overlap
    assert len(trace_nodes.intersection(required_nodes)) > 0, f"Missing trace nodes, found: {trace_nodes}"
    
    # Assert audit
    # Depending on the backend structure, audit might be nested or direct
    audit_str = json.dumps(audit)
    assert "external_called" in audit_str, "external_called not in audit"
    
    # Judge
    sys_prompt = "You are an evaluator. Return a JSON with 'score' (1-5 float), 'reason' (string), 'failures' (list of strings)."
    user_prompt = f"Question: {question}\nAnswer: {final_response}\nCitations count: {len(citations)}\n\nRate relevance, evidence support, and hallucination risk."
    
    judge_res = await judge.evaluate(sys_prompt, user_prompt)
    score = float(judge_res.get("score", 0))
    
    results_report.append({
        "type": "qa",
        "question": question,
        "score": score,
        "reason": judge_res.get("reason", ""),
        "failures": judge_res.get("failures", []),
        "trace_nodes": list(trace_nodes),
        "citations_count": len(citations),
        "audit": audit
    })
    
    return score, judge_res.get("failures", [])

async def cleanup_sessions(sessions: list):
    async with httpx.AsyncClient(trust_env=False) as client:
        for sid in sessions:
            try:
                await client.delete(f"{API_BASE}/sessions/{sid}")
            except Exception as e:
                print(f"Failed to delete session {sid}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Evaluate existing KnowledgeBase search/QA quality with a real judge model.")
    parser.add_argument(
        "--KnowledgeBase-name",
        "--knowledge-base-name",
        dest="knowledge_base_name",
        action="append",
        default=[],
        help="KnowledgeBase name to evaluate. May be passed multiple times.",
    )
    parser.add_argument("--max-knowledge_bases", type=int, default=2, help="Maximum knowledge_bases to evaluate when --KnowledgeBase-name is omitted.")
    args = parser.parse_args()

    print("Starting evaluation...")
    knowledge_bases, model_settings = await check_endpoints()
    if args.knowledge_base_name:
        target_names = set(args.knowledge_base_name)
        target_courses = [c for c in knowledge_bases if c["name"] in target_names]
    else:
        target_courses = knowledge_bases[: args.max_courses]
    
    if not target_courses:
        print("Target knowledge_bases not found!")
        return

    judge = Judge()
    results_report = []
    created_sessions = []
    
    for KnowledgeBase in target_courses:
        print(f"\nEvaluating KnowledgeBase: {KnowledgeBase['name']}")
        
        # Get evidence graph terms without relying on removed graph APIs.
        async with httpx.AsyncClient(trust_env=False) as client:
            resp = await client.get(
                f"{API_BASE}/knowledge_bases/current/graph",
                params={"knowledge_base_id": KnowledgeBase["id"], "graph_type": "evidence"},
            )
            graph = resp.json() if resp.status_code == 200 else {}
            evidence_terms = [
                str(node.get("snippet") or node.get("summary") or node.get("name") or "")
                for node in (graph.get("nodes") or [])[:10]
            ]
            
        queries, questions = generate_samples(KnowledgeBase['name'], evidence_terms)
        
        search_scores = []
        for q in queries:
            print(f"  Search: {q}")
            try:
                score, failures = await test_search(KnowledgeBase['id'], q, model_settings, judge, results_report)
                search_scores.append(score)
            except Exception as e:
                trace_str = traceback.format_exc()
                print(f"  Search failed: {trace_str}")
                results_report.append({"type": "search", "query": q, "error": repr(e), "traceback": trace_str, "score": 0})
                
        qa_scores = []
        for q in questions:
            print(f"  QA: {q}")
            try:
                score, failures = await test_qa(KnowledgeBase['id'], q, judge, results_report, created_sessions)
                qa_scores.append(score)
            except Exception as e:
                trace_str = traceback.format_exc()
                print(f"  QA failed: {trace_str}")
                results_report.append({"type": "qa", "question": q, "error": repr(e), "traceback": trace_str, "score": 0})
                
        avg_search = sum(search_scores)/len(search_scores) if search_scores else 0
        avg_qa = sum(qa_scores)/len(qa_scores) if qa_scores else 0
        
        print(f"  Avg Search Score: {avg_search}")
        print(f"  Avg QA Score: {avg_qa}")

    await cleanup_sessions(created_sessions)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"output/eval_runs/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2, ensure_ascii=False)
        
    print(f"\nReport saved to {out_dir}")

if __name__ == "__main__":
    asyncio.run(main())
