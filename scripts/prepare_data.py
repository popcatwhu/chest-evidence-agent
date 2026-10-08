"""Download source material and seed an explicitly synthetic workflow demonstration."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from chest_agent.config import DATA, ROOT
from chest_agent import store

sources = [
    ("pneumonia", "Pneumonia · 肺炎", "https://medlineplus.gov/pneumonia.html"),
    ("heartfailure", "Heart failure · 心力衰竭", "https://medlineplus.gov/heartfailure.html"),
    ("pleuraldisorders", "Pleural disorders · 胸膜疾病", "https://medlineplus.gov/pleuraldisorders.html"),
    ("pneumocystisinfections", "Pneumocystis infections · 肺孢子菌感染", "https://medlineplus.gov/pneumocystisinfections.html"),
]
for name,title,url in sources:
    response=requests.get(url,timeout=30)
    response.raise_for_status()
    soup=BeautifulSoup(response.text,"html.parser")
    summary=soup.select_one("#topic-summary")
    if summary is None:
        raise RuntimeError(f"未找到正文摘要：{url}")
    doc={"title":title,"url":url,"retrieved_at":datetime.now(timezone.utc).isoformat(),
         "type":"public health information, not a clinical guideline", "text":summary.get_text(" ",strip=True)}
    (DATA/"knowledge"/f"{name}.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    print('Knowledge:',name,len(doc['text']))

store.init_db()
if not any(c['title']=='教学演示 · 人工病史 + 公开样例胸片' for c in store.list_cases()):
    image=ROOT.parent/'MedRAX/demo/chest/pneumonia1.jpg'
    if not image.exists():
        raise RuntimeError('请先下载 MedRAX 官方样例图像')
    target=DATA/'uploads/demo.png'
    from PIL import Image
    Image.open(image).convert('RGB').save(target)
    case=store.create_case('教学演示 · 人工病史 + 公开样例胸片',
        '演示说明：以下病史为人工编写，不是该胸片的真实患者信息；仅测试系统流程，不能用于诊断准确率评估。\n'
        '年龄：45岁。\n主诉：咳嗽、发热3天。\n现病史：咳嗽伴少量痰，活动时气短。\n'
        '体温：38.5°C。\n既往史、血氧和实验室检查：未提供。\n'
        '图像来源：MedRAX 官方演示图像；该图像没有配套真实病史。',str(target))
    print('Demo case:',case['id'])
