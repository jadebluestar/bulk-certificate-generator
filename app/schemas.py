from datetime import date
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class Recipient(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: Optional[EmailStr] = None
    course_name: Optional[str] = Field(None, max_length=200)
    issue_date: Optional[str] = None

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be blank")
        # Reject control chars / injection-ish input for the renderer
        if any(ord(c) < 32 for c in v):
            raise ValueError("name contains control characters")
        return v


class JobCreateRequest(BaseModel):
    recipients: List[Recipient] = Field(..., min_length=1, max_length=10000)
    # Optional overrides for the template
    course_name: Optional[str] = None
    issue_date: Optional[str] = None


class CertificateOut(BaseModel):
    id: int
    recipient_name: str
    recipient_email: Optional[str]
    status: str
    error: Optional[str]
    download_url: Optional[str]

    class Config:
        from_attributes = True


class JobOut(BaseModel):
    id: int
    status: str
    total: int
    succeeded: int
    failed: int
    created_at: str
    updated_at: str
    message: Optional[str]

    class Config:
        from_attributes = True


class JobDetailOut(JobOut):
    certificates: List[CertificateOut]