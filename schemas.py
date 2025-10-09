from typing import List, Optional
from pydantic import BaseModel, Field

class Vulnerability(BaseModel):
    id: str
    title: str
    severity: str
    cvss: Optional[float] = None
    host: str
    port: Optional[str] = None
    protocol: Optional[str] = None
    description: Optional[str] = None
    recommendation: Optional[str] = None

class RemediationSuggestion(BaseModel):
    id: str
    proposed_commands: List[str] = Field(description="List of specific remediation commands for this vulnerability")
    notes: str = Field(description="Additional context or warnings about the remediation")

class BatchResult(BaseModel):
    suggestions: List[RemediationSuggestion] = Field(description="List of remediation suggestions for the vulnerabilities")
