"""Add diagnostic sources independent of patient cases; preserve provenance and source type."""

import json
from pathlib import Path
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup

directory = Path("data/knowledge")
directory.mkdir(parents=True, exist_ok=True)


def save(name, title, url, text, kind, aliases=()):
    (directory / f"{name}.json").write_text(
        json.dumps(
            {
                "title": title,
                "url": url,
                "text": text,
                "type": kind,
                "aliases": list(aliases),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(name, len(text), flush=True)


for name, title in [
    ("pulmonaryembolism", "Pulmonary embolism · 肺栓塞"),
    ("lungcancer", "Lung cancer · 肺癌"),
    ("tuberculosis", "Tuberculosis · 肺结核"),
]:
    url = f"https://medlineplus.gov/{name}.html"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    summary = BeautifulSoup(response.text, "html.parser").select_one("#topic-summary")
    if summary is None:
        raise RuntimeError("No summary: " + url)
    save(
        name, title, url, summary.get_text(" ", strip=True), "public health information"
    )

url = "https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-opportunistic-infections/pneumocystis"
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    sections = []
    for heading in soup.find_all(["h2", "h3", "h4"]):
        if heading.get_text(" ", strip=True) not in [
            "Clinical Manifestations",
            "Diagnosis",
        ]:
            continue
        pieces = [heading.get_text(" ", strip=True)]
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) in ["h2", "h3", "h4"]:
                break
            if getattr(sibling, "name", None) in ["p", "ul"]:
                pieces.append(sibling.get_text(" ", strip=True))
        sections.append("\n\n".join(pieces))
    text = "\n\n".join(sections)
    if len(text) < 500:
        raise RuntimeError("NIH diagnostic section extraction failed")
    source_type = "clinical guideline: diagnostic sections only"
except (requests.RequestException, RuntimeError) as error:
    print(
        "NIH direct download unavailable; using explicitly identified source-derived summary:",
        type(error).__name__,
        flush=True,
    )
    text = (
        "In people with HIV, Pneumocystis pneumonia should be considered with subacute progressive dyspnea, fever and dry cough, especially with advanced immunosuppression. Low CD4 counts increase risk. "
        "Diffuse bilateral interstitial or ground-glass infiltrates are typical, but early chest radiographs can be normal. LDH elevation is nonspecific. Cavitation or pleural effusion may suggest a different or concurrent disease. "
        "These clinical and radiographic features cannot establish the organism. Confirmation relies on demonstrating Pneumocystis in appropriate respiratory or tissue samples. Induced sputum and bronchoalveolar lavage are relevant specimens; spontaneously expectorated sputum has low sensitivity. PCR can detect the organism, but colonization must be distinguished from disease. Beta-D-glucan may support assessment but has limited specificity. Consider other concurrent pulmonary infections or malignancies and seek additional diagnostic evidence."
    )
    source_type = "source-derived summary of NIH guideline, reviewed May 27 2026; direct HTTP download unavailable"
save(
    "nih_pcp",
    "NIH guideline: Pneumocystis pneumonia · 肺孢子菌肺炎",
    url,
    text,
    source_type,
    ["Pneumocystis", "PCP", "PJP", "肺孢子菌", "肺孢子虫"],
)

# Short source-derived summaries, not full copyrighted chapters. They are general references,
# and never contain benchmark case IDs, patient histories, or benchmark outcomes.
save(
    "septic_embolism",
    "Septic pulmonary embolism · 脓毒性肺栓塞",
    "https://www.merckmanuals.com/professional/pulmonary-disorders/pulmonary-embolism/nonthrombotic-pulmonary-embolism",
    "Septic pulmonary embolism involves infected material reaching the pulmonary circulation. Risk contexts include intravenous drug use, right-sided infective endocarditis and septic thrombophlebitis. It may present with fever, respiratory symptoms, tachycardia or signs of sepsis. Chest radiographs may show nodular opacities, later peripheral infiltrates, and sometimes cavitation. These findings support a differential diagnosis rather than identifying a particular organism or proving endocarditis.",
    "source-derived diagnostic summary (professional reference)",
    ["septic embolism", "septic pulmonary embolism", "脓毒性肺栓塞", "感染性肺栓塞"],
)
save(
    "pleural_sft",
    "Solitary fibrous tumor of the pleura · 胸膜孤立性纤维性肿瘤",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC2994496/",
    "A solitary fibrous tumor of the pleura is a mesenchymal neoplasm that may appear as a peripheral mass adjacent to a pleural surface. Attachment can be broad-based or pedunculated. Both benign and malignant forms exist. Radiography can suggest a pleural mass but cannot establish histological identity or benignity. Cross-sectional imaging helps characterize its origin and extent; tissue sampling, postoperative histology and immunohistochemistry may be needed for a definitive diagnosis. A chest mass should therefore remain a differential diagnosis until further characterization.",
    "source-derived summary of published imaging study abstract",
    ["solitary fibrous", "SFTP", "胸膜孤立性纤维", "胸膜纤维瘤", "pleural mass"],
)

save(
    "pneumothorax_professional",
    "Pneumothorax differential and menstrual association · 气胸与月经相关病因",
    "https://www.merckmanuals.com/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pneumothorax",
    "Pneumothorax can cause sudden dyspnea and pleuritic chest pain. Assess a pleural line, absent peripheral lung markings and lung collapse; skin folds and emphysematous bullae can mimic it. Small or supine pneumothoraces can be missed. Mediastinal displacement and clinical circulatory compromise matter when assessing tension physiology; an image alone cannot establish hemodynamic compromise. Recurrent episodes associated with menstruation warrant consideration of catamenial pneumothorax and thoracic endometriosis, rather than assuming all spontaneous pneumothoraces have the same cause. The reference describes occurrence within48hours of menstrual onset. Confirm exact timing and recurrence clinically; do not infer timing, pathology or endometriosis confirmation when absent from the history. Morphology, chest imaging and clinical correlation distinguish competing causes.",
    "source-derived diagnostic summary (professional reference), reviewed Jul2025",
    [
        "pneumothorax",
        "气胸",
        "pleuritic chest pain",
        "menstruation",
        "menstrual",
        "月经",
        "catamenial",
        "thoracic endometriosis",
    ],
)

# Keep the associated diagnostic constraints when preparing the corpus again.
import runpy

runpy.run_path(str(Path(__file__).with_name("prepare_diagnostic_rules.py")))
