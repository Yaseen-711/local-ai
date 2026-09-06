"""Structured data specifications for industrial deliverables.

Provides Pydantic models for:
1. Technical Approval Note (DOCX)
2. Engineering Calculation Workbook (XLSX)
3. Executive / Board Presentation (PPTX)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 1. Technical Approval Note (DOCX)                                           #
# --------------------------------------------------------------------------- #

class ApprovalSignOff(BaseModel):
    """Formal sign-off entry for engineering approval."""
    role: str = Field(description="Role in engineering approval chain (e.g. Lead Process Engineer)")
    name: str = Field(description="Approver full name")
    title: str = Field(description="Professional title or designation")
    status: str = Field(default="PENDING", description="Sign-off status: 'APPROVED', 'PENDING', 'REJECTED'")
    date: Optional[str] = Field(default=None, description="Date of sign-off or action")
    comments: Optional[str] = Field(default=None, description="Engineering remarks or conditions")


class InspectionTagItem(BaseModel):
    """P&ID tag inspection or evaluation entry."""
    tag_id: str = Field(description="Equipment/instrument tag (e.g. FV-201A, HX-104)")
    description: str = Field(description="Equipment description or service line")
    pid_reference: str = Field(description="P&ID drawing reference number")
    service: str = Field(description="Fluid service or operating medium")
    design_spec: str = Field(description="Design specification rating or limits")
    measured_condition: str = Field(description="Observed inspection finding or parameter")
    compliance_status: str = Field(default="PASS", description="'PASS', 'FAIL', or 'ACTION_REQUIRED'")


class TechnicalApprovalNoteSpec(BaseModel):
    """Formal engineering technical approval note specification."""
    document_id: str = Field(description="Document reference number (e.g. ENG-NOTE-2026-088)")
    revision: str = Field(default="Rev 1.0", description="Document revision identifier")
    date: str = Field(description="Document date (e.g. 2026-09-06)")
    facility: str = Field(description="Plant or industrial facility name")
    unit_area: str = Field(description="Process unit or battery limit area")
    title: str = Field(description="Document subject or engineering title")
    author: str = Field(description="Authoring engineer or department")
    approver: Optional[str] = Field(default=None, description="Primary designated approval authority")
    status: str = Field(default="PENDING_APPROVAL", description="'APPROVED', 'PENDING_APPROVAL', 'FOR_REVIEW'")
    executive_summary: str = Field(description="Concise background, problem context, and justification")
    design_basis: str = Field(description="Governing codes, engineering standards, and process basis")
    operating_parameters: Dict[str, str] = Field(default_factory=dict, description="Key operating design parameters")
    inspection_findings: List[InspectionTagItem] = Field(default_factory=list, description="Tag inspection results")
    risk_assessment: List[Dict[str, str]] = Field(default_factory=list, description="Risk items, severity, mitigations")
    recommendations: List[str] = Field(default_factory=list, description="Numbered actionable recommendations")
    sign_offs: List[ApprovalSignOff] = Field(default_factory=list, description="Formal sign-off matrix")


# --------------------------------------------------------------------------- #
# 2. Engineering Calculation Workbook (XLSX)                                  #
# --------------------------------------------------------------------------- #

class ParameterDefinition(BaseModel):
    """Input design parameter for calculation workbook."""
    name: str = Field(description="Parameter description (e.g. Internal Design Pressure)")
    symbol: str = Field(description="Mathematical symbol (e.g. P)")
    cell_reference: str = Field(description="Spreadsheet cell coordinate (e.g. D7)")
    value: float = Field(description="Numerical parameter value")
    unit: str = Field(description="Unit of measurement (e.g. MPa, mm, °C)")
    source: str = Field(description="Document or standard source reference")
    tolerance: Optional[str] = Field(default=None, description="Tolerance or uncertainty range")


class CalculationStep(BaseModel):
    """Individual step in an engineering derivation with live Excel formula."""
    step_id: str = Field(description="Step identifier (e.g. '1.0', '1.1')")
    description: str = Field(description="Engineering description of the derivation step")
    symbol: str = Field(description="Output variable symbol (e.g. t_min, MAWP)")
    governing_equation: str = Field(description="Symbolic algebraic formula (e.g. '(P * D) / (2 * (S * E + P * Y)) + C')")
    substitution: str = Field(description="Equation with numerical substitutions (e.g. '(4.5 * 219.1) / ...')")
    excel_formula: str = Field(description="Live Excel formula string starting with '=' (e.g. '=(D7*D8)/(2*(D9*D10+D7*D11))+D12')")
    computed_value: float = Field(description="Numerically computed value")
    unit: str = Field(description="Engineering unit (e.g. 'mm', 'barg')")
    limit_reference: Optional[str] = Field(default=None, description="Code limit, threshold, or nominal value (e.g. 'Nominal Wall: 9.52 mm')")
    tolerance: Optional[str] = Field(default=None, description="Design tolerance range (e.g. '+15.0% / -12.5%')")
    verification_status: str = Field(default="PASS", description="Compliance verification status ('PASS', 'MARGINAL', 'FAIL')")
    status_formula: Optional[str] = Field(default=None, description="Live Excel formula for verification (e.g. '=IF(G8<=I8,\"PASS\",\"FAIL\")')")
    verification_evidence: Optional[str] = Field(default=None, description="Independent verification note or standard clause")


class VerificationEvidence(BaseModel):
    """Formal verification evidence separate from formula strings."""
    method: str = Field(description="Verification methodology (e.g. Independent Numerical Check & Code Audit)")
    verifier: str = Field(description="Name and title of independent verifying engineer")
    verification_date: str = Field(description="Verification date")
    status: str = Field(default="VERIFIED", description="'VERIFIED', 'CONDITIONAL', 'REJECTED'")
    evidence_notes: str = Field(description="Detailed verification observations, margin calculations, and notes")


class EngineeringCalculationSpec(BaseModel):
    """Engineering calculation workbook specification."""
    workbook_title: str = Field(description="Title of the calculation package")
    project_id: str = Field(description="Engineering project reference code")
    facility: str = Field(description="Industrial plant / facility name")
    author: str = Field(description="Authoring calculation engineer")
    checker: str = Field(description="Independent technical checker")
    date: str = Field(description="Calculation date")
    governing_standards: List[str] = Field(default_factory=list, description="Governing engineering standards")
    scope_description: str = Field(description="Scope and design objectives of the calculation")
    input_parameters: List[ParameterDefinition] = Field(default_factory=list, description="Design inputs table")
    steps: List[CalculationStep] = Field(default_factory=list, description="Derivation steps with live formulas")
    verification_evidence: Optional[VerificationEvidence] = Field(default=None, description="Separate verification evidence")
    conclusion: str = Field(description="Engineering conclusion and acceptability statement")


# --------------------------------------------------------------------------- #
# 3. Executive / Board Presentation (PPTX)                                    #
# --------------------------------------------------------------------------- #

class MetricCard(BaseModel):
    """Summary metric card for executive slides."""
    label: str = Field(description="Metric label (e.g. 'Total Tags Reviewed')")
    value: str = Field(description="Metric value (e.g. '42', '98.5%')")
    status: str = Field(default="NORMAL", description="'NORMAL', 'ALERT', 'CRITICAL'")


class PresentationSlideSpec(BaseModel):
    """Single presentation slide specification."""
    title: str = Field(description="Slide header title")
    subtitle: Optional[str] = Field(default=None, description="Slide subtitle or context")
    bullet_points: List[str] = Field(default_factory=list, description="Key bullet points")
    cards: List[MetricCard] = Field(default_factory=list, description="Metric cards to display")
    table_headers: List[str] = Field(default_factory=list, description="Table column headers")
    table_rows: List[List[str]] = Field(default_factory=list, description="Table rows")
    callout: Optional[str] = Field(default=None, description="Highlighted takeaway or decision callout")


class ExecutivePresentationSpec(BaseModel):
    """Executive / Board-style presentation specification."""
    presentation_title: str = Field(description="Main presentation title")
    presentation_subtitle: str = Field(description="Subtitle or board review topic")
    facility: str = Field(description="Industrial facility or business unit")
    presenter: str = Field(description="Presenting department or lead authority")
    date: str = Field(description="Presentation date (e.g. September 2026)")
    slides: List[PresentationSlideSpec] = Field(default_factory=list, description="Slide deck content")


# --------------------------------------------------------------------------- #
# Realistic Neutral Demo Deliverables Builders                                #
# --------------------------------------------------------------------------- #

def create_demo_approval_note() -> TechnicalApprovalNoteSpec:
    """Create a realistic neutral industrial Technical Approval Note."""
    return TechnicalApprovalNoteSpec(
        document_id="ENG-NOTE-2026-088",
        revision="Rev 1.0",
        date="2026-09-06",
        facility="Industrial Processing Complex",
        unit_area="Unit 200 - Fractionation & Desulfurization",
        title="Technical Integrity & Authorization Note: FV-201A Bypass Installation",
        author="Process Safety & Reliability Engineering Group",
        approver="Chief Operations Engineer / Engineering Authority",
        status="PENDING_APPROVAL",
        executive_summary=(
            "This Technical Approval Note evaluates the temporary bypass line modification for control valve "
            "FV-201A during planned on-stream maintenance. The technical assessment confirms that hydraulic capacity, "
            "pipe wall thickness, and relief margins satisfy governing process safety standards."
        ),
        design_basis="ASME B31.3 (Process Piping), API 520 Part I (Sizing & Selection of Pressure-Relieving Devices), ISO 14001",
        operating_parameters={
            "Design Pressure": "4.5 MPa (45.0 barg)",
            "Operating Temperature": "385 °C",
            "Normal Flow Rate": "145,000 kg/h",
            "Fluid Service": "Hydrocarbon Vapor / Condensate (Category D)",
            "Piping Material": "ASTM A106 Gr. B / Seamless Carbon Steel",
        },
        inspection_findings=[
            InspectionTagItem(
                tag_id="FV-201A",
                description="Fractionator Feed Control Valve",
                pid_reference="PID-AR-200-01, Rev 4",
                service="Hydrocarbon Feed",
                design_spec="Rating 300# RF, 45 barg, 385 °C",
                measured_condition="Trim erosion detected; bypass required for on-line overhaul",
                compliance_status="REQUIRES_ACTION",
            ),
            InspectionTagItem(
                tag_id="PSV-201",
                description="Column Top Thermal Relief Valve",
                pid_reference="PID-AR-200-01, Rev 4",
                service="Thermal Relief",
                design_spec="Set Pressure 48.0 barg, Orifice Size 4.5 cm²",
                measured_condition="Calibration valid through Nov 2026; relief capacity adequate",
                compliance_status="PASS",
            ),
            InspectionTagItem(
                tag_id="PI-203",
                description="Upstream Manifold Pressure Transmitter",
                pid_reference="PID-AR-200-02, Rev 3",
                service="Instrumentation",
                design_spec="0 - 60 barg, 4-20mA HART",
                measured_condition="Reading 41.2 barg; verified against local test gauge",
                compliance_status="PASS",
            ),
        ],
        risk_assessment=[
            {
                "Risk Event": "Overpressurization during manual valve switchover",
                "Severity": "High",
                "Probability": "Low",
                "Mitigation Measure": "Staged crack-open procedure with two qualified operators and continuous DCS pressure monitoring",
                "Residual Risk": "Low",
            },
            {
                "Risk Event": "Thermal expansion stress on bypass spool",
                "Severity": "Medium",
                "Probability": "Low",
                "Mitigation Measure": "Flexibility analysis completed in calculation workbook CALC-PR-2026-104 confirming stresses within ASME B31.3 limits",
                "Residual Risk": "Low",
            },
        ],
        recommendations=[
            "1. Authorize Management of Change (MOC) MOC-ENG-2026-088 for temporary bypass operation.",
            "2. Implement standard lock-out/tag-out (LOTO) isolation on FV-201A upstream and downstream block valves.",
            "3. Enforce maximum continuous operating limit of 42.0 barg during bypass mode.",
            "4. Complete overhaul of FV-201A within the scheduled 72-hour operational window.",
        ],
        sign_offs=[
            ApprovalSignOff(
                role="Lead Process Safety Engineer",
                name="A. R. Mitchell, P.E.",
                title="Lead Process Safety Specialist",
                status="APPROVED",
                date="2026-09-06",
                comments="Technical safety review complete; procedures comply with plant safety requirements.",
            ),
            ApprovalSignOff(
                role="Operations Area Manager",
                name="S. K. Rao",
                title="Operations Superintendent - Area 200",
                status="PENDING",
                date=None,
                comments="Awaiting final shift handover briefing before signing.",
            ),
            ApprovalSignOff(
                role="Chief Engineering Authority",
                name="H. E. Vance, Ph.D.",
                title="VP Engineering & Technical Services",
                status="PENDING",
                date=None,
                comments="Final approval pending operational sign-off.",
            ),
        ],
    )


def create_demo_calculation_workbook() -> EngineeringCalculationSpec:
    """Create a realistic neutral Engineering Calculation Workbook with live formulas."""
    return EngineeringCalculationSpec(
        workbook_title="Piping Wall Thickness & Relief Margin Verification",
        project_id="CALC-PR-2026-104",
        facility="Industrial Processing Complex",
        author="Lead Mechanical Systems Engineer",
        checker="Principal Piping & Pressure Systems Auditor",
        date="2026-09-06",
        governing_standards=[
            "ASME B31.3-2022: Process Piping Code (Section 304.1.2)",
            "API Standard 520 Part I (10th Ed.): Sizing and Selection of Pressure-Relieving Devices",
            "ASTM A106 / ASME SA106: Specification for Seamless Carbon Steel Pipe",
        ],
        scope_description=(
            "Verification of minimum required pipe wall thickness, maximum allowable working pressure (MAWP), "
            "and thermal relief safety margin for the Unit 200 FV-201A bypass spool (NPS 8 Sched 40 Seamless Carbon Steel)."
        ),
        input_parameters=[
            ParameterDefinition(
                name="Internal Design Pressure",
                symbol="P",
                cell_reference="D7",
                value=4.5,
                unit="MPa",
                source="Process Design Specification PDS-200-01",
                tolerance="±0.1 MPa",
            ),
            ParameterDefinition(
                name="Pipe Outside Diameter",
                symbol="D",
                cell_reference="D8",
                value=219.1,
                unit="mm",
                source="ASME B36.10M (NPS 8 Nominal)",
                tolerance="±0.75 mm",
            ),
            ParameterDefinition(
                name="Basic Allowable Stress",
                symbol="S",
                cell_reference="D9",
                value=137.9,
                unit="MPa",
                source="ASME B31.3 Table A-1 (ASTM A106 Gr B @ 385°C)",
                tolerance="Code Minimum",
            ),
            ParameterDefinition(
                name="Longitudinal Quality Factor",
                symbol="E",
                cell_reference="D10",
                value=1.0,
                unit="dimensionless",
                source="ASME B31.3 Table 302.3.4 (Seamless)",
                tolerance="Exact",
            ),
            ParameterDefinition(
                name="Wall Thickness Coefficient",
                symbol="Y",
                cell_reference="D11",
                value=0.4,
                unit="dimensionless",
                source="ASME B31.3 Table 304.1.1 (Ferritic Steel @ 385°C)",
                tolerance="Exact",
            ),
            ParameterDefinition(
                name="Mechanical & Corrosion Allowance",
                symbol="C",
                cell_reference="D12",
                value=1.5,
                unit="mm",
                source="Plant Piping Material Specification PMS-CS-01",
                tolerance="Minimum Allowance",
            ),
            ParameterDefinition(
                name="Nominal Pipe Wall Thickness",
                symbol="t_nom",
                cell_reference="D13",
                value=9.52,
                unit="mm",
                source="ASME B36.10M (NPS 8 Sched 40)",
                tolerance="-12.5% mill tolerance (8.33 mm min)",
            ),
            ParameterDefinition(
                name="Mill Under-Tolerance Margin",
                symbol="MT",
                cell_reference="D14",
                value=0.875,
                unit="dimensionless",
                source="ASTM A106 Section 16.2 (12.5% allowance)",
                tolerance="Code Standard",
            ),
        ],
        steps=[
            CalculationStep(
                step_id="1.0",
                description="Minimum Pressure Design Wall Thickness (ASME B31.3 Eq 3a)",
                symbol="t_calc",
                governing_equation="t_calc = (P * D) / (2 * (S * E + P * Y))",
                substitution="t_calc = (4.5 * 219.1) / (2 * (137.9 * 1.0 + 4.5 * 0.4))",
                excel_formula="=(D7*D8)/(2*(D9*D10+D7*D11))",
                computed_value=3.53,
                unit="mm",
                limit_reference="Base Pressure Design Limit",
                tolerance="Theoretical Minimum",
                verification_status="PASS",
                status_formula='=IF(G8>0,"PASS","FAIL")',
                verification_evidence="Math derivation confirmed: 985.95 / 279.4 = 3.5288 mm.",
            ),
            CalculationStep(
                step_id="1.1",
                description="Total Required Minimum Thickness including Corrosion Allowance",
                symbol="t_min",
                governing_equation="t_min = t_calc + C",
                substitution="t_min = 3.53 + 1.50",
                excel_formula="=G8+D12",
                computed_value=5.03,
                unit="mm",
                limit_reference="Minimum Code Limit (with C)",
                tolerance="±0.05 mm",
                verification_status="PASS",
                status_formula='=IF(G9<=D13*D14,"PASS","FAIL")',
                verification_evidence="ASME B31.3 Eq 3a + C = 5.0288 mm. Exceeds minimum required thickness.",
            ),
            CalculationStep(
                step_id="1.2",
                description="Minimum Available Thickness after Mill Tolerance (t_avail = t_nom * MT)",
                symbol="t_avail",
                governing_equation="t_avail = t_nom * MT",
                substitution="t_avail = 9.52 * 0.875",
                excel_formula="=D13*D14",
                computed_value=8.33,
                unit="mm",
                limit_reference="Must exceed t_min (5.03 mm)",
                tolerance="8.33 mm",
                verification_status="PASS",
                status_formula='=IF(G10>=G9,"PASS","FAIL")',
                verification_evidence="Available thickness 8.33 mm exceeds t_min 5.03 mm by 3.30 mm (65.6% margin).",
            ),
            CalculationStep(
                step_id="2.0",
                description="Maximum Allowable Working Pressure (MAWP) for Pipe Spool",
                symbol="MAWP",
                governing_equation="MAWP = (2 * S * E * (t_avail - C)) / (D - 2 * Y * (t_avail - C))",
                substitution="MAWP = (2 * 137.9 * 1.0 * (8.33 - 1.5)) / (219.1 - 2 * 0.4 * (8.33 - 1.5))",
                excel_formula="=(2*D9*D10*(G10-D12))/(D8-2*D11*(G10-D12))",
                computed_value=8.81,
                unit="MPa",
                limit_reference="Design Pressure: 4.50 MPa",
                tolerance="Upper Bound Limit",
                verification_status="PASS",
                status_formula='=IF(G11>=D7,"PASS","FAIL")',
                verification_evidence="MAWP = 8.81 MPa (88.1 barg), providing 95.8% margin above design pressure 4.50 MPa.",
            ),
            CalculationStep(
                step_id="3.0",
                description="Thermal Relief Capacity Safety Factor (API 520)",
                symbol="SF_relief",
                governing_equation="SF_relief = Q_rated / Q_required",
                substitution="SF_relief = 1250 / 840",
                excel_formula="=1250/840",
                computed_value=1.49,
                unit="ratio",
                limit_reference="Minimum Code Safety Factor: 1.10",
                tolerance=">= 1.10",
                verification_status="PASS",
                status_formula='=IF(G12>=1.10,"PASS","FAIL")',
                verification_evidence="Relief safety factor 1.49 exceeds mandatory 1.10 minimum margin required by API 520 Part I.",
            ),
        ],
        verification_evidence=VerificationEvidence(
            method="Independent Numerical Verification & ASME B31.3 Section 304.1.2 Audit",
            verifier="Principal Piping & Pressure Systems Auditor, P.E.",
            verification_date="2026-09-06",
            status="VERIFIED",
            evidence_notes=(
                "All formula derivations independently cross-checked against ASME B31.3-2022 Section 304.1.2 and API 520. "
                "Available pipe thickness of 8.33 mm provides 65.6% safety margin over required 5.03 mm. "
                "MAWP of 8.81 MPa provides 95.8% over-pressure headroom above design pressure 4.50 MPa. "
                "Calculation package is certified sound for operational use."
            ),
        ),
        conclusion=(
            "The proposed NPS 8 Sched 40 ASTM A106 Gr. B bypass spool fully satisfies all wall thickness, "
            "allowable stress, and thermal relief safety requirements under ASME B31.3 and API 520. "
            "Design is verified and approved for temporary on-stream bypass operation."
        ),
    )


def create_demo_executive_presentation() -> ExecutivePresentationSpec:
    """Create a realistic neutral Executive / Board-style presentation."""
    return ExecutivePresentationSpec(
        presentation_title="Process Safety & Integrity Review: Unit 200 Bypass",
        presentation_subtitle="Executive Engineering Review Board Decision Memorandum",
        facility="Industrial Processing Complex - Area 200",
        presenter="Process Engineering & Technical Safety Directorate",
        date="September 2026",
        slides=[
            PresentationSlideSpec(
                title="Executive Summary & Context",
                subtitle="Operational justification for temporary bypass installation",
                bullet_points=[
                    "Fractionator Feed Control Valve FV-201A exhibited internal trim erosion during continuous operations.",
                    "On-line bypass installation enables overhaul of FV-201A without unplanned shutdown of Unit 200.",
                    "Rigorous engineering verification confirms zero compromise to process containment or plant safety margins.",
                ],
                cards=[
                    MetricCard(label="Design Pressure", value="4.5 MPa", status="NORMAL"),
                    MetricCard(label="Pipe Thickness Margin", value="+65.6%", status="NORMAL"),
                    MetricCard(label="Relief Capacity Factor", value="1.49x", status="NORMAL"),
                    MetricCard(label="Required Downtime", value="72 Hours", status="NORMAL"),
                ],
                callout="Key Takeaway: Bypass installation preserves continuous production while satisfying all ASME B31.3 safety margins.",
            ),
            PresentationSlideSpec(
                title="P&ID & Equipment Inspection Findings",
                subtitle="Detailed integrity assessment across affected equipment tags",
                bullet_points=[
                    "Three primary equipment tags evaluated under Management of Change MOC-ENG-2026-088.",
                    "Direct visual and ultrasound inspection confirms adjacent piping integrity is sound.",
                ],
                table_headers=["Tag ID", "Description", "Service", "Measured Finding", "Status"],
                table_rows=[
                    ["FV-201A", "Feed Control Valve", "Hydrocarbon", "Trim erosion; bypass required", "ACTION"],
                    ["PSV-201", "Thermal Relief Valve", "Thermal Relief", "Set pressure 48 barg valid", "PASS"],
                    ["PI-203", "Pressure Transmitter", "Instrumentation", "Reading 41.2 barg calibrated", "PASS"],
                ],
                callout="Only valve FV-201A requires intervention; all safety relief systems are fully operational.",
            ),
            PresentationSlideSpec(
                title="Engineering Calculations & Verification",
                subtitle="ASME B31.3 & API 520 code compliance derivations",
                bullet_points=[
                    "Wall thickness calculated per ASME B31.3 Section 304.1.2 Eq. 3a with 1.5 mm corrosion allowance.",
                    "Piping MAWP evaluated at 8.81 MPa, exceeding 4.5 MPa design operating pressure by 95.8%.",
                    "Independent technical check certified by Principal Piping Engineer.",
                ],
                table_headers=["Calculation Step", "Symbol", "Calculated Value", "Code Limit", "Verification"],
                table_rows=[
                    ["Required Wall Thickness", "t_min", "5.03 mm", "<= 8.33 mm", "PASS (+65.6%)"],
                    ["Available Wall Thickness", "t_avail", "8.33 mm", "Nominal 9.52 mm", "PASS (Code Compliant)"],
                    ["Piping MAWP", "MAWP", "8.81 MPa", ">= 4.50 MPa", "PASS (+95.8%)"],
                    ["Thermal Relief Margin", "SF_relief", "1.49 ratio", ">= 1.10 ratio", "PASS (+35.5%)"],
                ],
            ),
            PresentationSlideSpec(
                title="Risk Mitigation & Operational Controls",
                subtitle="Defense-in-depth safety measures during 72-hour bypass mode",
                bullet_points=[
                    "Standard lock-out/tag-out (LOTO) procedures enforced on valve isolation blocks.",
                    "Two qualified process operators stationed on-site during switchover with live DCS interlocks.",
                    "Continuous ultrasonic monitoring scheduled every 12 hours across the bypass spool.",
                    "Contingency shutdown procedure prepared if upstream manifold pressure exceeds 42.0 barg.",
                ],
                cards=[
                    MetricCard(label="Residual Safety Risk", value="Low", status="NORMAL"),
                    MetricCard(label="Shift Audit Frequency", value="Every 12h", status="NORMAL"),
                    MetricCard(label="Emergency Tripping", value="Active", status="NORMAL"),
                ],
            ),
            PresentationSlideSpec(
                title="Decision Requested from the Board",
                subtitle="Authorization recommendation and immediate implementation timeline",
                bullet_points=[
                    "1. Authorize Management of Change (MOC) MOC-ENG-2026-088 for temporary bypass operation.",
                    "2. Approve 72-hour maintenance window commencing Saturday 06:00 UTC.",
                    "3. Authorize procurement of replacement trim set for FV-201A from certified OEM supplier.",
                ],
                cards=[
                    MetricCard(label="Estimated Budget", value="$18,500", status="NORMAL"),
                    MetricCard(label="Production Loss Avoided", value="$340,000", status="NORMAL"),
                    MetricCard(label="Safety Incident Risk", value="Zero", status="NORMAL"),
                ],
                callout="Recommendation: Board grants formal approval to proceed with MOC-ENG-2026-088.",
            ),
        ],
    )
