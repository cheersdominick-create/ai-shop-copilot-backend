from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os
from openai import OpenAI

# ------------------------
# App Initialization
# ------------------------

app = FastAPI(title="AI Shop Co-Pilot")

# CORS (Permanent Correct Setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-domain.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5-mini"

# ------------------------
# Data Models
# ------------------------

class DiagnosticRequest(BaseModel):
    year: int
    make: str
    model: str
    engine: str
    mileage: int
    obd_codes: List[str]
    customer_complaint: str
    tech_observations: str


class LikelyCause(BaseModel):
    rank: int
    cause: str
    why_likely: str
    confirmation_tests: List[str]


class DiagnosticResponse(BaseModel):
    summary: str
    likely_causes: List[LikelyCause]
    common_pitfalls: List[str]
    disclaimer: str


class PaperworkResponse(BaseModel):
    technician_notes: str
    customer_explanation: str
    repair_order_summary: str


class PaperworkFromDiagnosisRequest(BaseModel):
    vehicle: DiagnosticRequest
    diagnosis: DiagnosticResponse


class RunAllResponse(BaseModel):
    diagnosis: DiagnosticResponse
    paperwork: PaperworkResponse

# ------------------------
# Internal Logic
# ------------------------

def run_diagnosis(vehicle: DiagnosticRequest) -> DiagnosticResponse:
    prompt = f"""
You are an automotive diagnostic assistant helping a professional repair shop.

Vehicle:
{vehicle.year} {vehicle.make} {vehicle.model} {vehicle.engine}
Mileage: {vehicle.mileage}

OBD-II Codes:
{', '.join(vehicle.obd_codes)}

Customer Complaint:
{vehicle.customer_complaint}

Technician Observations:
{vehicle.tech_observations}

INSTRUCTIONS:
- Provide diagnostic guidance only
- Do NOT provide a definitive diagnosis
- Rank likely causes
- Provide confirmation tests
- Avoid guarantees or repair instructions

Respond ONLY in valid JSON using this schema:

{{
  "summary": string,
  "likely_causes": [
    {{
      "rank": number,
      "cause": string,
      "why_likely": string,
      "confirmation_tests": [string]
    }}
  ],
  "common_pitfalls": [string],
  "disclaimer": string
}}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": "You assist automotive technicians with diagnostic reasoning only."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        parsed = json.loads(response.output_text)
        return DiagnosticResponse(**parsed)
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid diagnostic AI response")


def generate_paperwork(vehicle: DiagnosticRequest, diagnosis: DiagnosticResponse) -> PaperworkResponse:
    prompt = f"""
You are assisting an automotive repair shop with documentation.
You are NOT diagnosing the vehicle.

Vehicle:
{vehicle.year} {vehicle.make} {vehicle.model} {vehicle.engine}, {vehicle.mileage} miles

Customer Complaint:
{vehicle.customer_complaint}

Technician Observations:
{vehicle.tech_observations}

DIAGNOSTIC SUMMARY:
{diagnosis.summary}

LIKELY CAUSES:
{chr(10).join([f"{c.rank}. {c.cause}" for c in diagnosis.likely_causes])}

INSTRUCTIONS:
Generate:
1) Technician internal notes
2) Customer-friendly explanation
3) Repair order summary

Respond ONLY in valid JSON using this schema:

{{
  "technician_notes": string,
  "customer_explanation": string,
  "repair_order_summary": string
}}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": "You generate professional automotive repair documentation."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        parsed = json.loads(response.output_text)
        return PaperworkResponse(**parsed)
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid paperwork AI response")

# ------------------------
# Routes
# ------------------------

@app.get("/")
def health_check():
    return {"status": "AI Shop Co-Pilot running"}


@app.post("/diagnose", response_model=DiagnosticResponse)
def diagnose_vehicle(vehicle: DiagnosticRequest):
    return run_diagnosis(vehicle)


@app.post("/paperwork", response_model=PaperworkResponse)
def paperwork_from_diagnosis(request: PaperworkFromDiagnosisRequest):
    return generate_paperwork(request.vehicle, request.diagnosis)


@app.post("/run-all", response_model=RunAllResponse)
def run_all(vehicle: DiagnosticRequest):
    diagnosis = run_diagnosis(vehicle)
    paperwork = generate_paperwork(vehicle, diagnosis)

    return {
        "diagnosis": diagnosis,
        "paperwork": paperwork
    }
