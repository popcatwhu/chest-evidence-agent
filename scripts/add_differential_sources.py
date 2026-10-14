"""Fetch professional diagnostic references, independently of evaluation labels."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    (
        "pap_professional",
        "Pulmonary alveolar proteinosis · 肺泡蛋白沉积症",
        "interstitial-lung-diseases/pulmonary-alveolar-proteinosis",
        ["pulmonary alveolar proteinosis", "肺泡蛋白沉积症", "肺泡蛋白沉着症"],
    ),
    (
        "abpa_professional",
        "Allergic bronchopulmonary aspergillosis · 变应性支气管肺曲霉病",
        "asthma-and-related-disorders/allergic-bronchopulmonary-aspergillosis-abpa",
        ["allergic bronchopulmonary aspergillosis", "ABPA", "曲霉", "曲菌"],
    ),
    (
        "sarcoidosis_professional",
        "Sarcoidosis · 结节病",
        "sarcoidosis/sarcoidosis",
        ["sarcoidosis", "结节病"],
    ),
]


def fetch_sources():
    records = []
    for name, title, path, aliases in SOURCES:
        url = "https://www.merckmanuals.com/professional/pulmonary-disorders/" + path
        response = requests.get(url, timeout=35)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.select_one("main")
        if main is None or not main.select_one("h1"):
            raise ValueError("Professional reference body missing")
        # Exclude navigation, scripts and footers; paragraph/headings/list items
        # retain diagnostic language without an LLM rewriting the source.
        blocks, seen = [], set()
        for node in main.select("h1,h2,h3,p,li"):
            if node.find_parent(["nav", "footer", "script", "style"]):
                continue
            text = node.get_text(" ", strip=True)
            if len(text) >= 25 and text not in seen:
                blocks.append(text)
                seen.add(text)
        text = "\n\n".join(blocks)
        if len(text) < 1000 or "Diagnosis" not in text:
            raise ValueError("Diagnostic reference extraction failed")
        record = {
            "title": title,
            "url": url,
            "text": text,
            "type": "professional clinical reference; not a patient-specific confirmation",
            "aliases": aliases,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "html_sha256": hashlib.sha256(response.content).hexdigest(),
        }
        destination = ROOT / "data/knowledge" / (name + ".json")
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        records.append(
            {
                k: record[k]
                for k in ["title", "url", "type", "retrieved_at", "html_sha256"]
            }
        )
        print(name, len(text), flush=True)
    (ROOT / "evals/differential-sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    fetch_sources()
