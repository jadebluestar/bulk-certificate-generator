"""
FastAPI application for the Bulk Certificate Generator.

Endpoints:
    GET  /health
    POST /jobs                          -> create a bulk generation job
    GET  /jobs/{job_id}                 -> full job detail (with certificates)
    GET  /jobs/{job_id}/status          -> lightweight progress/status
    GET  /jobs/{job_id}/certificates    -> list certificates for a job
    GET  /certificates/{cert_id}        -> download a generated certificate PNG

Design notes:
    - POST /jobs returns 202 Accepted immediately. Actual rendering happens
      in the background worker (app.worker). The client polls status.
    - Validation is delegated to Pydantic (app.schemas). Invalid requests
      fail fast with 422 and never create a Job row.
    - Failure isolation lives in the worker, not here: one bad recipient
      cannot prevent other certificates in the same job from being generated.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import csv
import io
from fastapi import File, UploadFile
from typing import Optional

from app import worker
from app.database import Base, engine, get_db
from app.models import Certificate, CertStatus, Job, JobStatus
from app.schemas import (
    CertificateOut,
    JobCreateRequest,
    JobDetailOut,
    JobOut,
)

# ---------------------------------------------------------------------------
# App + DB bootstrap
# ---------------------------------------------------------------------------

# Create tables on startup. In a real deployment you'd use Alembic migrations,
# but for this project create_all is sufficient and keeps setup trivial.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Bulk Certificate Generator",
    version="1.0.0",
    description=(
        "Submit a bulk certificate generation request, track progress, and "
        "retrieve generated certificates."
    ),
)


@app.on_event("startup")
async def _on_startup() -> None:
    """Start the in-process background worker."""
    worker.start_worker()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Gracefully stop the background worker."""
    await worker.stop_worker()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.post(
    "/jobs",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
    summary="Create a bulk certificate generation job",
)
async def create_job(
    payload: JobCreateRequest,
    db: Session = Depends(get_db),
) -> JobOut:
    """
    Accept a list of recipients and enqueue a background generation job.

    Returns immediately with a job id. The client should poll
    GET /jobs/{id}/status (or GET /jobs/{id}) until the status is one of
    completed / partial / failed.
    """
    # Create the job row first so we have an id to attach certificates to.
    job = Job(
        status=JobStatus.PENDING,
        total=len(payload.recipients),
        succeeded=0,
        failed=0,
        message=None,
    )
    db.add(job)
    db.flush()  # assigns job.id without committing

    # Default issue_date: request-level if provided, else today.
    default_issue_date = (
        payload.issue_date or datetime.utcnow().strftime("%Y-%m-%d")
    )

    # One Certificate row per recipient. These start PENDING; the worker
    # flips them to SUCCESS or FAILED.
    for recipient in payload.recipients:
        db.add(
            Certificate(
                job_id=job.id,
                recipient_name=recipient.name,
                recipient_email=str(recipient.email) if recipient.email else None,
                course_name=recipient.course_name or payload.course_name,
                issue_date=recipient.issue_date or default_issue_date,
                status=CertStatus.PENDING,
            )
        )

    db.commit()
    db.refresh(job)

    # Hand off to the background worker. If the queue is unavailable,
    # mark the job as failed instead of leaving it dangling in PENDING.
    try:
        await worker.enqueue_job(job.id)
    except Exception as exc:  # pragma: no cover - defensive
        job.status = JobStatus.FAILED
        job.message = f"Failed to enqueue job: {type(exc).__name__}: {exc}"
        job.updated_at = datetime.utcnow()
        db.commit()
        raise HTTPException(
            status_code=500, detail="Failed to enqueue job for processing"
        )

    return _job_to_out(job)

@app.post(
    "/jobs/upload",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
    summary="Create a job by uploading a CSV/Excel file of recipients",
)
async def create_job_from_file(
    file: UploadFile = File(..., description="CSV or XLSX with recipient rows"),
    course_name: Optional[str] = None,
    issue_date: Optional[str] = None,
    db: Session = Depends(get_db),
) -> JobOut:
    """
    Accept a file with recipient rows. Expected columns (case-insensitive,
    order-independent):
        name          (required)
        email         (optional)
        course_name   (optional, overrides the query param)
        issue_date    (optional, overrides the query param)

    Extra columns are ignored. Blank rows are skipped. Rows with an invalid
    name are collected and reported as a 422 with details, so the client
    knows exactly which rows to fix.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty")

    filename = (file.filename or "").lower()
    try:
        rows = _parse_uploaded_file(raw, filename)
    except Exception as exc:
        raise HTTPException(
            400, f"Could not parse file: {type(exc).__name__}: {exc}"
        )

    if not rows:
        raise HTTPException(400, "No data rows found in file")

    # Validate each row through Pydantic so behavior matches the JSON endpoint.
    valid_recipients: list[Recipient] = []
    errors: list[dict] = []

    for i, row in enumerate(rows, start=2):  # start=2 because row 1 is header
        try:
            valid_recipients.append(Recipient(**row))
        except Exception as exc:
            # Pydantic ValidationError -> concise per-row error
            msg = str(exc).splitlines()[0] if str(exc) else "invalid row"
            errors.append({"row": i, "error": msg, "data": row})

    if not valid_recipients:
        raise HTTPException(
            422,
            detail={"message": "No valid rows found", "row_errors": errors},
        )

    # Reuse the same job creation path as JSON submissions.
    payload = JobCreateRequest(
        recipients=valid_recipients,
        course_name=course_name,
        issue_date=issue_date,
    )
    # Note: we deliberately do NOT fail the whole job if some rows were bad.
    # Valid rows are processed; the response message reports the skipped rows.
    job_out = await create_job(payload, db=db)

    if errors:
        # Attach a note to the job so the client can see partial validation issues.
        job = db.query(Job).filter(Job.id == job_out.id).first()
        if job:
            job.message = f"Skipped {len(errors)} invalid row(s): " + \
                          "; ".join(f"row {e['row']}: {e['error']}" for e in errors)
            db.commit()
            db.refresh(job)
            job_out = _job_to_out(job)

    return job_out


@app.get(
    "/jobs/{job_id}",
    response_model=JobDetailOut,
    tags=["jobs"],
    summary="Get full job detail including per-certificate status",
)
def get_job(job_id: int, db: Session = Depends(get_db)) -> JobDetailOut:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    certs: List[Certificate] = (
        db.query(Certificate)
        .filter(Certificate.job_id == job_id)
        .order_by(Certificate.id.asc())
        .all()
    )

    base = _job_to_out(job)
    return JobDetailOut(
        **base.model_dump(),
        certificates=[_cert_to_out(c) for c in certs],
    )


@app.get(
    "/jobs/{job_id}/status",
    response_model=JobOut,
    tags=["jobs"],
    summary="Lightweight job progress/status",
)
def get_job_status(job_id: int, db: Session = Depends(get_db)) -> JobOut:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_out(job)


@app.get(
    "/jobs/{job_id}/certificates",
    tags=["jobs"],
    summary="List certificates for a job (with download URLs)",
)
def list_job_certificates(job_id: int, db: Session = Depends(get_db)) -> dict:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    certs: List[Certificate] = (
        db.query(Certificate)
        .filter(Certificate.job_id == job_id)
        .order_by(Certificate.id.asc())
        .all()
    )

    succeeded = sum(1 for c in certs if c.status == CertStatus.SUCCESS)
    failed = sum(1 for c in certs if c.status == CertStatus.FAILED)

    return {
        "job_id": job_id,
        "job_status": _enum_value(job.status),
        "total": len(certs),
        "succeeded": succeeded,
        "failed": failed,
        "pending": len(certs) - succeeded - failed,
        "certificates": [_cert_to_out(c).model_dump() for c in certs],
    }


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------

@app.get(
    "/certificates/{cert_id}",
    tags=["certificates"],
    summary="Download a generated certificate PNG",
)
def download_certificate(cert_id: int, db: Session = Depends(get_db)):
    cert = db.query(Certificate).filter(Certificate.id == cert_id).first()
    if cert is None:
        raise HTTPException(status_code=404, detail="Certificate not found")

    if cert.status != CertStatus.SUCCESS or not cert.file_path:
        raise HTTPException(
            status_code=409,
            detail=f"Certificate not ready (status={_enum_value(cert.status)})",
        )

    if not os.path.exists(cert.file_path):
        # Row says success but the file is gone (disk cleanup, etc.)
        raise HTTPException(
            status_code=410, detail="Certificate file missing on disk"
        )

    return FileResponse(
        cert.file_path,
        media_type="image/png",
        filename=f"certificate_{cert.id}.png",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum_value(value) -> str:
    """Return the string value of an Enum, or str(value) if already a string."""
    return value.value if hasattr(value, "value") else str(value)


def _job_to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        status=_enum_value(job.status),
        total=job.total,
        succeeded=job.succeeded,
        failed=job.failed,
        created_at=job.created_at.isoformat() if job.created_at else "",
        updated_at=job.updated_at.isoformat() if job.updated_at else "",
        message=job.message,
    )


def _cert_to_out(cert: Certificate) -> CertificateOut:
    is_success = cert.status == CertStatus.SUCCESS and cert.file_path
    return CertificateOut(
        id=cert.id,
        recipient_name=cert.recipient_name,
        recipient_email=cert.recipient_email,
        status=_enum_value(cert.status),
        error=cert.error,
        download_url=f"/certificates/{cert.id}" if is_success else None,
    )

def _parse_uploaded_file(raw: bytes, filename: str) -> list[dict]:
    """
    Return a list of dicts, one per data row, with normalized keys:
        {"name": str, "email": str|None, "course_name": str|None, "issue_date": str|None}
    Raises on unparseable input.
    """
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        import pandas as pd
        df = pd.read_excel(io.BytesIO(raw), dtype=str)
    elif filename.endswith(".csv") or filename.endswith(".txt") or filename == "":
        # Try utf-8, fall back to latin-1 for Excel-exported CSVs
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        # Sniff the dialect so we handle ; or \t delimiters too
        try:
            dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        df = __import__("pandas").DataFrame(list(reader))
    else:
        raise ValueError(f"Unsupported file type: {filename!r}. Use .csv or .xlsx")

    # Normalize column names: lowercase, strip, spaces->underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Accept some common aliases
    aliases = {
        "recipient_name": "name",
        "full_name": "name",
        "student_name": "name",
        "participant_name": "name",
        "mail": "email",
        "e-mail": "email",
        "course": "course_name",
        "date": "issue_date",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    if "name" not in df.columns:
        raise ValueError(
            f"No 'name' column found. Got columns: {list(df.columns)}"
        )

    # Keep only the columns we care about, drop fully-empty rows.
    keep = [c for c in ["name", "email", "course_name", "issue_date"]
            if c in df.columns]
    df = df[keep].dropna(how="all")

    out: list[dict] = []
    for _, row in df.iterrows():
        record = {}
        for col in keep:
            val = row.get(col)
            if val is None:
                continue
            # pandas puts NaN for empty cells; treat as missing
            if isinstance(val, float) and val != val:
                continue
            s = str(val).strip()
            if s == "":
                continue
            record[col] = s
        # Skip rows that are entirely empty after cleaning
        if record:
            out.append(record)
    return out