import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"   # some succeeded, some failed
    FAILED = "failed"     # all failed


class CertStatus(str, enum.Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    total = Column(Integer, default=0)
    succeeded = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message = Column(Text, nullable=True)

    certificates = relationship(
        "Certificate", back_populates="job", cascade="all, delete-orphan"
    )


class Certificate(Base):
    __tablename__ = "certificates"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    recipient_name = Column(String, nullable=False)
    recipient_email = Column(String, nullable=True)
    course_name = Column(String, nullable=True)
    issue_date = Column(String, nullable=True)
    status = Column(Enum(CertStatus), default=CertStatus.PENDING, nullable=False)
    file_path = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    job = relationship("Job", back_populates="certificates")