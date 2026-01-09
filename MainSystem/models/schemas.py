from dataclasses import dataclass
from typing import Optional

@dataclass
class Claim:
    character: str
    action: str
    time: Optional[str]
    location: Optional[str]

@dataclass
class Requirement:
    capability: str
    must_hold_before: Optional[str]
    confidence: float = 1.0

@dataclass
class Event:
    subject: str            # who did it
    relation: str           # what happened
    object: Optional[str]   # to whom / what
    time: Optional[str]
    location: Optional[str]
    confidence: float = 1.0