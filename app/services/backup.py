import csv
import io
import os
from datetime import datetime, timezone
from typing import Iterable, Optional, List, Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import create_engine, text

from app.core.config import settings


def _get_engine_url() -> str:
    return settings.DATABASE_URL


def _export_table_to_csv_bytes(engine_url: str, table_name: str) -> bytes:
    engine = create_engine(engine_url)
    with engine.connect() as conn:
        res = conn.execute(text(f"SELECT * FROM {table_name}"))
        output = io.StringIO()
        writer = csv.writer(output)
        if res.returns_rows:
            writer.writerow(res.keys())
            for row in res:
                writer.writerow(list(row))
        return output.getvalue().encode("utf-8")


def _s3_client():
    # Standard AWS env vars should be configured in Render/host if S3 is desired
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    return boto3.client("s3")


def backup_database_to_s3(prefix: Optional[str] = None) -> Dict[str, object]:
    """Export key tables to CSV and upload to S3.

    Env vars required:
      - DB_BACKUP_S3_BUCKET: destination bucket
      - optional DB_BACKUP_S3_PREFIX: default prefix if not provided

    Returns a dict with uploaded object keys.
    """
    bucket = os.environ.get("DB_BACKUP_S3_BUCKET")
    if not bucket:
        raise RuntimeError("DB_BACKUP_S3_BUCKET not set")
    prefix = prefix or os.environ.get("DB_BACKUP_S3_PREFIX", "db-backups")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engine_url = _get_engine_url()

    tables: List[str] = [
        "users",
        "foods",
        "custom_foods",
        "food_logs",
        "sport_activities",
        "daily_summaries",
        "weight_logs",
    ]

    s3 = _s3_client()
    uploaded: List[str] = []
    for table in tables:
        csv_bytes = _export_table_to_csv_bytes(engine_url, table)
        key = f"{prefix}/{now}/{table}.csv"
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
            uploaded.append(key)
        except (BotoCoreError, ClientError) as e:
            # best-effort: continue with other tables
            continue
    return {"bucket": bucket, "keys": uploaded}


def backup_database_locally(directory: Optional[str] = None) -> Dict[str, object]:
    """Fallback: write CSVs to local disk inside the container.

    Default directory: /app/backups/YYYYMMDD
    """
    base_dir = directory or "/app/backups"
    now_day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = os.path.join(base_dir, now_day)
    os.makedirs(out_dir, exist_ok=True)

    engine_url = _get_engine_url()
    tables: List[str] = [
        "users",
        "foods",
        "custom_foods",
        "food_logs",
        "sport_activities",
        "daily_summaries",
        "weight_logs",
    ]
    written: List[str] = []
    for table in tables:
        csv_bytes = _export_table_to_csv_bytes(engine_url, table)
        path = os.path.join(out_dir, f"{table}.csv")
        with open(path, "wb") as f:
            f.write(csv_bytes)
        written.append(path)
    return {"directory": out_dir, "files": written}


def perform_backup() -> Dict[str, object]:
    """Try S3; fall back to local. Returns a summary dict."""
    try:
        bucket = os.environ.get("DB_BACKUP_S3_BUCKET")
        if bucket:
            return {"mode": "s3", **backup_database_to_s3()}
    except Exception as e:
        # fall back to local
        pass
    return {"mode": "local", **backup_database_locally()}


