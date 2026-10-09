import asyncio
import traceback
from datetime import datetime
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Job, Certificate, JobStatus, CertStatus
from app.certificate import generate_certificate

_queue: "asyncio.Queue[int]" = asyncio.Queue()
_worker_task: asyncio.Task | None = None


async def _process_job(job_id: int):
    db: Session = SessionLocal()
    try:
        job = db.query(Job).get(job_id)
        if not job:
            return
        job.status = JobStatus.PROCESSING
        job.updated_at = datetime.utcnow()
        db.commit()

        certs = db.query(Certificate).filter(Certificate.job_id == job_id).all()
        succeeded = 0
        failed = 0

        for cert in certs:
            try:
                # Simulate CPU-bound rendering without blocking event loop
                loop = asyncio.get_running_loop()
                path = await loop.run_in_executor(
                    None,
                    generate_certificate,
                    cert.recipient_name,
                    cert.course_name,
                    cert.issue_date,
                )
                cert.status = CertStatus.SUCCESS
                cert.file_path = path
                cert.error = None
                succeeded += 1
            except Exception as e:
                cert.status = CertStatus.FAILED
                cert.error = f"{type(e).__name__}: {e}"
                failed += 1

            job.succeeded = succeeded
            job.failed = failed
            job.updated_at = datetime.utcnow()
            db.commit()

        if failed == 0:
            job.status = JobStatus.COMPLETED
        elif succeeded == 0:
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.PARTIAL
        job.updated_at = datetime.utcnow()
        db.commit()
    except Exception:
        job = db.query(Job).get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.message = traceback.format_exc()
            job.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


async def _worker_loop():
    while True:
        job_id = await _queue.get()
        try:
            await _process_job(job_id)
        finally:
            _queue.task_done()


def start_worker():
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())


async def stop_worker():
    global _worker_task
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None


async def enqueue_job(job_id: int):
    await _queue.put(job_id)