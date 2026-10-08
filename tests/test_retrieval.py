import json
from chest_agent import tools


def test_retrieval_limits_one_source_and_keeps_links(tmp_path,monkeypatch):
    knowledge=tmp_path/'knowledge'
    knowledge.mkdir()
    for name,content in [('one','pneumonia fever cough '*400),('two','pneumonia fever cough differential '*100)]:
        (knowledge/f'{name}.json').write_text(json.dumps({'title':name,'text':content,'url':f'https://example.org/{name}'}))
    monkeypatch.setattr(tools,'DATA',tmp_path)
    results=tools.retrieve('pneumonia fever cough')
    assert sum(e.source=='https://example.org/one' for e in results)<=2
    assert any(e.source=='https://example.org/two' for e in results)
