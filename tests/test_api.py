import io
from PIL import Image
from fastapi.testclient import TestClient
from chest_agent import store
from chest_agent.app import app


def test_upload_supplement_and_input_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "test.sqlite3")
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    with TestClient(app) as client:
        response = client.post(
            "/api/cases",
            data={"title": "测试病例", "context": "原始资料"},
            files={"image": ("image.png", image.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        case_id = response.json()["id"]
        run_id = store.create_run(case_id, "question", "direct")
        response = client.post(
            f"/api/cases/{case_id}/supplement", json={"text": "新检查结果"}
        )
        assert "新检查结果" in response.json()["context"]
        assert store.get_run(run_id)["context"] == "原始资料"
        assert client.get(f"/api/cases/{case_id}/image").status_code == 200


def test_model_not_ready_does_not_fake_result(tmp_path, monkeypatch):
    from chest_agent.llm import backend

    monkeypatch.setattr(store, "DB", tmp_path / "test.sqlite3")
    monkeypatch.setattr(backend, "ready", lambda: False)
    with TestClient(app) as client:
        case = client.post(
            "/api/cases", data={"title": "测试", "context": "待分析"}
        ).json()
        response = client.post(f"/api/cases/{case['id']}/runs", json={})
        assert response.status_code == 503


def test_benchmark_pause_keeps_history_accessible_and_rejects_new_gpu_jobs(
    tmp_path, monkeypatch
):
    from chest_agent import config
    from chest_agent.llm import backend

    monkeypatch.setattr(store, "DB", tmp_path / "test.sqlite3")
    monkeypatch.setattr(config, "INFERENCE_PAUSED", True)
    monkeypatch.setattr(backend, "ready", lambda: True)
    with TestClient(app) as client:
        case = client.post(
            "/api/cases", data={"title": "已有病例", "context": "病史"}
        ).json()
        assert client.get("/api/cases").status_code == 200
        assert client.get("/api/health").json()["inference_paused"]
        response = client.post(f"/api/cases/{case['id']}/runs", json={})
        assert response.status_code == 503
        assert client.get(f"/api/cases/{case['id']}/runs").json() == []
