"""Exercise the actual UI and LAN gateway; use --inference for one real resident-model job."""

import argparse
import json
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()
    output = ROOT / "data/frontend"
    output.mkdir(exist_ok=True)
    known = json.loads(
        (
            ROOT / "data/medical_comparison/lingshu32_clinical_regression.json"
        ).read_text()
    )
    example_id = known["records"][0]["run"]["case_id"]
    examples = json.loads((ROOT / "data/quality/cases.json").read_text())
    image = ROOT / "data" / examples[0]["image_path"]
    errors, failures, checks, created = [], [], [], []
    smoke_run = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(
                viewport={"width": 1440, "height": 1000}, device_scale_factor=1
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "response",
                lambda response: (
                    failures.append(f"{response.status} {response.url}")
                    if response.status >= 400 and response.url.startswith(args.url)
                    else None
                ),
            )
            page.goto(args.url, wait_until="networkidle")
            expect(page.locator("#health-label")).to_have_text("服务已连接")
            assert page.locator(".case-item").count() > 0
            checks.append("LAN page, assets, health and case library")
            page.screenshot(path=str(output / "desktop-empty.png"), full_page=True)

            page.locator("#case-title").fill("前端自动验证 · " + str(time.time_ns()))
            page.locator("#case-context-input").fill(
                "用于界面上传和保存验证，不提交模型推理。"
            )
            page.locator("#image-upload").set_input_files(str(image))
            expect(page.locator("#preview")).to_be_visible()
            page.locator("#image-open").click()
            expect(page.locator("#image-dialog")).to_be_visible()
            page.keyboard.press("Escape")
            with page.expect_response(
                lambda r: r.url.endswith("/api/cases") and r.request.method == "POST"
            ) as event:
                page.locator("#save-case").click()
            case = event.value.json()
            created.append(case["id"])
            expect(page.locator("#case-detail")).to_be_visible()
            expect(page.locator("#preview")).to_have_attribute(
                "src", f"/api/cases/{case['id']}/image"
            )
            checks.append("Real multipart upload, image preview, dialog and save")

            page.locator("#supplement-section summary").click()
            page.locator("#supplement").fill("新增界面验证资料。")
            page.locator("#save-supplement").click()
            expect(page.locator("#case-context")).to_contain_text("新增界面验证资料。")
            checks.append("Supplement saved through real API")
            page.locator("#top-new-case").click()
            page.locator("#case-title").fill("前端无胸片验证 · " + str(time.time_ns()))
            page.locator("#case-context-input").fill(
                "仅用于检查无图病例不显示上一例胸片。"
            )
            with page.expect_response(
                lambda r: r.url.endswith("/api/cases") and r.request.method == "POST"
            ) as event:
                page.locator("#save-case").click()
            created.append(event.value.json()["id"])
            expect(page.locator("#case-detail")).to_be_visible()
            expect(page.locator("#image-section")).to_be_hidden()
            checks.append("Image-free case clears previous image")

            page.goto(args.url + "?case=" + example_id, wait_until="networkidle")
            expect(page.locator(".primary-diagnosis h2")).to_be_visible()
            page.screenshot(path=str(output / "desktop-report.png"), full_page=True)
            page.locator(".primary-diagnosis [data-ref]").first.click()
            expect(page.locator("#evidence-content")).to_be_visible()
            expect(page.locator(".evidence-item.highlight")).to_be_visible()
            checks.append("Citation opens and highlights actual evidence")
            page.locator("#tab-history").click()
            expect(page.locator(".history-item").first).to_be_visible()
            page.locator(".history-item").last.click()
            expect(page.locator("#report-content")).to_be_visible()
            expect(page.locator(".primary-diagnosis")).to_be_visible()
            checks.append("Historical report switching")
            with page.expect_download() as download:
                page.locator("#download-report").click()
            download.value.save_as(str(output / "downloaded-report.json"))
            assert json.loads((output / "downloaded-report.json").read_text())["result"]
            checks.append("Complete report JSON download")
            page.locator("#image-open").click()
            page.keyboard.press("Escape")
            page.locator("#connection-button").click()
            expect(page.locator("#access-url")).to_have_value(args.url)
            page.locator("#copy-url").click()
            expect(page.locator("#toast")).to_have_text("访问地址已复制")
            page.keyboard.press("Escape")
            checks.append("Connection dialog and LAN clipboard fallback")

            page.locator('[data-view="evaluation"]').click()
            expect(page.locator(".eval-table tbody tr")).to_have_count(5)
            expect(page.locator(".eval-stat").nth(1)).to_contain_text("75%")
            page.screenshot(path=str(output / "evaluation.png"), full_page=True)
            checks.append("Real frozen evaluation snapshot")

            page.set_viewport_size({"width": 390, "height": 844})
            page.locator("#menu-toggle").click()
            expect(page.locator("#sidebar")).to_have_class("sidebar open")
            page.locator('[data-view="workspace"]').click()
            expect(page.locator("#sidebar")).not_to_have_class("sidebar open")
            page.wait_for_function(
                'document.getElementById("sidebar").getBoundingClientRect().right <= 1'
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(path=str(output / "mobile-report.png"), full_page=True)
            checks.append("390px mobile drawer and no horizontal overflow")

            if args.inference:
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.locator('[data-mode="verified"]').click()
                page.locator("#question").fill(
                    "请综合原始胸片和已知病史，给出最可能诊断和有依据的鉴别诊断。"
                )
                with page.expect_response(
                    lambda r: "/runs" in r.url and r.request.method == "POST"
                ) as event:
                    page.locator("#analyze").click()
                smoke_run = event.value.json()["run_id"]
                page.wait_for_function(
                    'localStorage.getItem("chestActiveRun") !== null'
                )
                expect(page.locator("#progress-panel")).to_be_visible()
                page.reload(wait_until="networkidle")
                page.wait_for_function(
                    'localStorage.getItem("chestActiveRun") === null', timeout=600000
                )
                expect(page.locator("#run-status")).to_have_text("已完成")
                expect(page.locator(".primary-diagnosis")).to_be_visible()
                page.screenshot(
                    path=str(output / "inference-complete.png"), full_page=True
                )
                checks.append(
                    "Real GPU inference, refresh recovery and completed report"
                )
            assert not errors, errors
            assert not failures, failures
            browser.close()
    finally:
        # Remove only the cases generated by this UI test; keep the real inference record.
        from chest_agent import store

        for case_id in created:
            case = store.get_case(case_id)
            if not case or not case["title"].startswith(
                ("前端自动验证 · ", "前端无胸片验证 · ")
            ):
                continue
            with store.connect() as db:
                if db.execute(
                    "SELECT 1 FROM runs WHERE case_id=?", (case_id,)
                ).fetchone():
                    continue
                db.execute("DELETE FROM cases WHERE id=?", (case_id,))
            if case["image_path"]:
                path = Path(case["image_path"])
                if path.parent == ROOT / "data/uploads":
                    path.unlink(missing_ok=True)
        (output / "browser_validation.json").write_text(
            json.dumps(
                {
                    "url": args.url,
                    "checks": checks,
                    "page_errors": errors,
                    "failed_responses": failures,
                    "real_inference_run": smoke_run,
                    "test_cases_cleaned_up": created,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(
        json.dumps(
            {"checks": checks, "page_errors": errors, "real_inference_run": smoke_run},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
