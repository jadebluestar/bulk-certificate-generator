
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
````

**macOS:**

```bash
brew install tesseract
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the Default Certificate Template

Generate the default certificate template once before starting the server.

```bash
python templates/make_template.py
```

### 5. Start the Server

```bash
python run.py
```

The server runs at:

* **API Base URL:** [http://localhost:8000](http://localhost:8000)
* **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Usage

### 1. Submit a Certificate Generation Request

Send a `POST` request to `/jobs` with the recipient details, course name, and issue date.

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "recipients": [
      {
        "name": "Alice Johnson",
        "email": "alice@example.com"
      },
      {
        "name": "Bob Smith",
        "email": "bob@example.com"
      }
    ],
    "course_name": "Advanced Python",
    "issue_date": "2024-06-01"
  }'
```

### Example Response

The API returns **HTTP 202 Accepted**, indicating that the request has been accepted for background processing.

```json
{
  "id": 1,
  "status": "pending",
  "total": 2,
  "succeeded": 0,
  "failed": 0
}
```

The response fields indicate:

| Field       | Description                                   |
| ----------- | --------------------------------------------- |
| `id`        | Unique identifier for the generation job      |
| `status`    | Current processing status                     |
| `total`     | Total number of certificates requested        |
| `succeeded` | Number of successfully generated certificates |
| `failed`    | Number of failed certificate generations      |

The exact response fields depend on the API schema.

### 2. Check Job Progress

Retrieve the current job status.

```bash
curl http://localhost:8000/jobs/1/status
```

You can also retrieve the job details using:

```bash
curl http://localhost:8000/jobs/1
```

#### Job Statuses

| Status       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `pending`    | The job has been accepted and is waiting to be processed.    |
| `processing` | Certificate generation is in progress.                       |
| `completed`  | All requested certificates were generated successfully.      |
| `partial`    | Some certificates succeeded while others failed.             |
| `failed`     | The job failed without successfully generating certificates. |

A job is marked `partial` when at least one certificate succeeds and at least one fails.

### 3. Retrieve Generated Certificates

List the certificates associated with a job.

```bash
curl http://localhost:8000/jobs/1/certificates
```

The response provides information about the certificates generated for the specified job.

### 4. Download an Individual Certificate

Download a certificate using its certificate ID.

```bash
curl -O http://localhost:8000/certificates/1
```

The downloaded file is a PNG image of the generated certificate.

---

## Design Decisions

### 1. Asynchronous Background Processing

Generating certificates for thousands of recipients synchronously can cause HTTP timeouts and tie up application workers.

To avoid this, the API returns an HTTP `202 Accepted` response with a job ID immediately after accepting a valid request.

Certificate generation then runs in the background through an in-process asynchronous worker.

**Current implementation:**

* Accepts bulk generation requests.
* Creates a job to track the request.
* Processes recipients in the background.
* Records individual certificate results.
* Updates job progress and status.

**Production considerations:**

The current worker runs within the application process. For production deployments, it can be replaced with a dedicated task queue such as Celery or RQ backed by Redis.

A dedicated queue would improve durability, scalability, and worker management. The current in-process implementation should not be treated as a durable distributed job queue.

### 2. ML-Based Text Placement

**Module:** `app/placement.py`

One of the main challenges is determining where a recipient's name should appear on a certificate without hardcoding pixel coordinates for every template.

The placement engine combines image processing, OCR, machine learning, and deterministic fallbacks.

#### Approach

**Step 1: Detect Candidate Text Regions**

The certificate template is processed as an image. Maximally Stable Extremal Regions (MSER) and Tesseract OCR are used to identify candidate text regions.

**Step 2: Extract Features**

Each candidate region is represented by a feature vector containing properties such as:

* Normalized centroid coordinates.
* Region dimensions and size.
* Aspect ratio.
* Detected text length.
* OCR confidence.
* Whether the region is horizontally centered.
* Whether the region lies within the middle vertical band.
* Whether the region resembles a placeholder.

**Step 3: Train a Random Forest Classifier**

A Random Forest classifier is trained using synthetic certificate templates.

Training labels are derived from a structural heuristic: the recipient's name is expected to be the largest centered text region within the middle vertical band.

The model learns to recognize regions matching these characteristics.

**Step 4: Predict the Name Region**

During inference, the model scores the detected regions and selects the candidate most likely to contain the recipient's name.

**Step 5: Apply a Deterministic Fallback**

If the prediction confidence is low, the engine falls back to a centered-band heuristic.

If the template contains an explicit `{{name}}` placeholder, its bounding box is used directly, bypassing the machine learning pipeline.

#### Why This Approach?

The system uses machine learning as a generalization layer rather than relying entirely on a trained model.

* Explicit placeholders provide deterministic placement.
* The classifier helps identify likely name regions in templates without explicit anchors.
* A heuristic fallback provides an alternative when the model is uncertain.
* The model is trained lazily on first use and cached for subsequent requests.

The cached model is stored at:

```text
models/placement_model.joblib
```

**Note:** The classifier's performance depends on the quality and diversity of its synthetic training data and the characteristics of the input templates. The fallback improves resilience but does not guarantee correct placement on every arbitrary certificate design.

### 3. Per-Recipient Failure Isolation

Each certificate is generated independently within its own `try/except` block in the background worker.

If a recipient's certificate cannot be generated:

1. The failure is caught and recorded.
2. The corresponding certificate database record stores the error information.
3. Processing continues for the remaining recipients.
4. The job status reflects the final combination of successful and failed certificates.

This design prevents an individual rendering failure from unnecessarily terminating an entire bulk job.

### 4. Request Validation

The API uses Pydantic schemas to validate incoming requests.

Validation includes:

* Non-blank recipient names.
* Well-formed email addresses.
* Recipient list size limits.
* Required request fields and expected data types.

Invalid requests are rejected with HTTP `422 Unprocessable Entity` before a generation job is created.

### 5. Database Design

The application uses SQLite for simplicity.

Two primary tables track the generation workflow:

| Table          | Purpose                                                              |
| -------------- | -------------------------------------------------------------------- |
| `jobs`         | Stores job metadata, status, and generation progress.                |
| `certificates` | Stores recipient information, generation results, and error details. |

Each job can contain multiple certificate records, with one record corresponding to each recipient.

The database connection is managed through SQLAlchemy.

**Production considerations:**

For deployments requiring concurrent database access, higher write throughput, or multiple application instances, SQLite can be replaced with PostgreSQL by configuring the database connection appropriately.

---

## Project Structure

```text
.
├── app/
│   ├── main.py          # FastAPI routes and API endpoints
│   ├── models.py        # SQLAlchemy ORM models
│   ├── schemas.py       # Pydantic request and response schemas
│   ├── database.py      # Database engine and session management
│   ├── worker.py        # Asynchronous background processor
│   ├── certificate.py   # Certificate image rendering with PIL
│   └── placement.py     # ML-based text placement engine
│
├── templates/
│   └── make_template.py # Default certificate template generator
│
├── output/              # Generated certificate PNG files
├── models/              # Cached trained ML model
├── tests/               # Automated test suite
├── requirements.txt     # Python dependencies
└── run.py               # Application entry point
```

---

## Running Tests

Run the automated test suite using pytest.

```bash
pytest -v tests/
```

The test suite can be used to verify API behavior, request validation, job processing, and certificate generation.

---

## Extending the Project

### 1. Support Multiple Certificate Templates

Add a template identifier or template path to the generation request.

The application can then resolve the requested template and pass it to the certificate generation function.

The placement engine can process each template independently.

### 2. Add PDF Output

The current implementation generates PNG certificates using Pillow (PIL).

PDF output can be introduced by modifying the rendering pipeline to save the certificate in PDF format.

For example:

```python
image.save(output_path, "PDF")
```

The image dimensions and resolution should be configured appropriately for the intended PDF page size.

### 3. Introduce a Distributed Task Queue

Replace the in-process background worker with Celery or RQ backed by Redis.

This enables dedicated workers to process jobs independently of the API server.

Additional production improvements could include:

* Retry policies for transient failures.
* Configurable concurrency.
* Task timeouts.
* Queue monitoring.
* Durable job execution.
* Rate limiting and resource management.

### 4. Improve the Placement Model

The current model uses synthetic templates and heuristic-derived labels.

Its accuracy can potentially be improved by:

1. Collecting representative real-world certificate templates.
2. Manually labeling the correct name regions.
3. Expanding the training dataset.
4. Evaluating predictions on templates excluded from training.
5. Retraining the classifier.

The cached model can be regenerated using:

```python
train_placement_model(force=True)
```

This assumes the training function is exposed by `app/placement.py`.

---

## Technology Stack

| Technology    | Purpose                                   |
| ------------- | ----------------------------------------- |
| Python        | Application development                   |
| FastAPI       | REST API framework                        |
| Pydantic      | Request validation and schema definitions |
| SQLAlchemy    | Database ORM                              |
| SQLite        | Job and certificate metadata storage      |
| Pillow (PIL)  | Certificate image generation              |
| Tesseract OCR | Text recognition                          |
| MSER          | Candidate text-region detection           |
| scikit-learn  | Random Forest classification              |
| Joblib        | Machine learning model persistence        |
| asyncio       | In-process asynchronous processing        |
| pytest        | Automated testing                         |

---

## Quick Start

Run the following commands on Ubuntu/Debian to set up the project.

### Install the system dependency

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Set up Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate the template

```bash
python templates/make_template.py
```

### Start the server

```bash
python run.py
```

### Run the tests

In another terminal, activate the virtual environment and run:

```bash
pytest -v tests/
```

### Explore the API

Open the interactive API documentation:

[http://localhost:8000/docs](http://localhost:8000/docs)

---
Even added a template generator, which can be customized

