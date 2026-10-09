# Bulk Certificate Generator

Generate personalized certificates for thousands of recipients in a single API request.

## Features

- **Bulk API** — Generate certificates for thousands of recipients in a single request.
- **Asynchronous Background Processing** — Returns a `job_id` immediately while certificates are generated in the background.
- **ML-Based Text Placement** — Automatically identifies where the recipient's name belongs on different certificate templates.
- **Per-Recipient Failure Isolation** — A failure for one recipient does not interrupt certificate generation for others.
- **Progress Tracking** — Monitor job progress using `/jobs/{id}/status` or `/jobs/{id}`.
- **Download Endpoints** — Retrieve individual certificate PNG files.
- **Request Validation** — Validates recipient names, email addresses, and request limits before creating a job.
- **Persistent Job Tracking** — Stores job and certificate metadata in SQLite.
- **Interactive API Documentation** — Explore and test endpoints through FastAPI's Swagger UI.

---

## Setup

### Prerequisites

- Python 3.10+
- Tesseract OCR
- pip

### 1. Install Tesseract OCR

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr