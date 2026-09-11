import os, json
from dotenv import load_dotenv
load_dotenv()

def ai_analyze(profile, ml):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "summary": "AI is not configured. Add OPENAI_API_KEY to .env to enable generated insights.",
            "insights": [
                "Review columns with high missing-value percentages.",
                "Investigate duplicate rows before modeling.",
                "Review strong correlations and detected anomalies."
            ]
        }
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        compact = {
            "rows": profile["rows"], "columns": profile["columns_count"],
            "quality_score": profile["quality_score"],
            "duplicates": profile["duplicates"],
            "columns": profile["columns"],
            "correlations": profile["correlations"][:15],
            "ml": ml,
        }
        prompt = (
            "Act as a senior data scientist. Analyze this structured CSV profile. "
            "Return concise JSON with keys summary, key_findings, data_quality_risks, "
            "recommendations, and modeling_notes. Do not invent facts.\n\n"
            + json.dumps(compact, default=str)
        )
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text
        try:
            parsed = json.loads(text)
            return {"enabled": True, **parsed}
        except Exception:
            return {"enabled": True, "summary": text, "key_findings": [], "data_quality_risks": [], "recommendations": [], "modeling_notes": []}
    except Exception as exc:
        return {"enabled": False, "summary": f"AI request failed: {exc}", "insights": []}
