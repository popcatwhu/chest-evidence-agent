import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
from .config import DATA

DB = DATA / "cases.sqlite3"


def connect():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, context TEXT NOT NULL,
          image_path TEXT, created TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY, case_id TEXT NOT NULL, context TEXT NOT NULL,
          question TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
          created TEXT NOT NULL, result TEXT, error TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
          stage TEXT NOT NULL, message TEXT NOT NULL, created TEXT NOT NULL
        );
        """)


def now():
    return datetime.now(timezone.utc).isoformat()


def create_case(title, context, image_path=None):
    case_id = uuid4().hex
    with connect() as db:
        db.execute("INSERT INTO cases VALUES (?,?,?,?,?)", (case_id, title, context, image_path, now()))
    return get_case(case_id)


def get_case(case_id):
    with connect() as db:
        row = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return dict(row) if row else None


def append_context(case_id, text):
    with connect() as db:
        db.execute("UPDATE cases SET context=context || ? WHERE id=?", ("\n补充资料：" + text, case_id))


def list_cases():
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM cases ORDER BY created DESC")]


def create_run(case_id, question, mode):
    case = get_case(case_id)
    run_id = uuid4().hex
    with connect() as db:
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                   (run_id, case_id, case["context"], question, mode, "queued", now(), None, None))
    return run_id


def update_run(run_id, status, result=None, error=None):
    with connect() as db:
        db.execute("UPDATE runs SET status=?,result=?,error=? WHERE id=?",
                   (status, json.dumps(result, ensure_ascii=False) if result is not None else None, error, run_id))


def event(run_id, stage, message):
    with connect() as db:
        db.execute("INSERT INTO events(run_id,stage,message,created) VALUES(?,?,?,?)", (run_id, stage, message, now()))


def get_run(run_id):
    with connect() as db:
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["events"] = [dict(e) for e in db.execute("SELECT * FROM events WHERE run_id=? ORDER BY seq", (run_id,))]
        return result


def previous_report(case_id, current_run):
    with connect() as db:
        row = db.execute("SELECT result FROM runs WHERE case_id=? AND id!=? AND status='completed' ORDER BY created DESC LIMIT 1",
                         (case_id, current_run)).fetchone()
    return json.loads(row["result"])["report"] if row and row["result"] else None


def recover_interrupted():
    with connect() as db:
        db.execute("UPDATE runs SET status='interrupted',error='服务重启，任务中断；请重新分析' WHERE status IN ('queued','running')")


def previous_context(case_id,current_run):
    with connect() as db:
        row=db.execute("SELECT context FROM runs WHERE case_id=? AND id!=? AND status='completed' ORDER BY created DESC LIMIT 1",
            (case_id,current_run)).fetchone()
    return row['context'] if row else None
