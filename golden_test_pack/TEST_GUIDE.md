MRPL GOLDEN E2E TEST PACK
============================

All documents in this pack are SYNTHETIC and exist only for validation.

FILES
-----
RAG corpus (ingest these 3):
  01_equipment_spec.pdf
  02_inspection_report.pdf
  03_operating_manual.pdf

Direct inputs:
  05_pid_direct_input.png
  04_direct_only_datasheet.pdf

IMPORTANT:
  Do NOT ingest 04_direct_only_datasheet.pdf into RAG when testing direct-only document QA.

GOLDEN TESTS
------------

1) RAG-only
Ask:
"What is the design pressure of FV-201A, and what material is specified?"

Expected facts:
  FV-201A
  45.0 barg
  316L Stainless Steel

Expected route:
  Decision/Planning -> retrieval.rag -> reranking -> grounded QA

2) RAG cross-document
Ask:
"What are the design pressure, normal operating pressure, and latest inspection condition of FV-201A?"

Expected facts:
  Design pressure: 45.0 barg
  Normal operating pressure: 38.0 barg
  Inspection: 0.12 mm corrosion; routine maintenance attention recommended

Expected route:
  retrieval.rag across multiple documents -> grounded QA

3) Direct PDF ONLY
Upload 04_direct_only_datasheet.pdf but do not ingest it.
Ask:
"What is the design pressure and asset tag in this document?"

Expected:
  HX-104
  42.5 barg

Expected route:
  direct document understanding/QA, NOT RAG

4) Image + RAG
Provide 05_pid_direct_input.png.
Ask:
"Identify FV-201A in this P&ID and tell me its design pressure according to the equipment specification."

Expected:
  Vision identifies FV-201A
  RAG supplies 45.0 barg
  Answer cites the specification source

Expected route:
  vision.inspect + retrieval.rag + synthesis

5) RAG -> XLSX
Ask:
"Using the knowledge base, prepare an engineering calculation workbook for FV-201A showing the design pressure, operating pressure, corrosion measurement, and verification status."

Expected:
  Uses RAG evidence
  artifact.generate -> XLSX
  formulas/status preserved
  artifact hash/download available

6) RAG -> PDF/DOCX
Ask:
"Using the knowledge base, create a technical approval note for FV-201A summarizing the inspection condition and pressure limits."

Expected route:
  retrieval.rag -> grounded evidence -> artifact.generate -> DOCX/PDF

7) RAG -> PPTX
Ask:
"Using the knowledge base, create an executive summary presentation for FV-201A."

Expected route:
  retrieval.rag -> grounded evidence -> artifact.generate -> PPTX

8) Code/calculation
Ask:
"Using the engineering values for FV-201A, calculate a specified engineering quantity and verify the calculation with executable code."

Expected route:
  decision/planning -> code.verify_and_repair -> code.workspace sandbox
  Host execution must NOT occur.

9) Multi-capability
Provide 05_pid_direct_input.png and ask:
"Review this P&ID against the equipment information in the knowledge base, identify any discrepancy, perform the relevant engineering verification, and create an executive summary."

Do NOT prescribe capability order.
Verify the system chooses an appropriate composition of:
  vision.inspect
  retrieval.rag
  code.workspace / verification
  artifact.generate

10) Negative grounding
Ask:
"What is the design pressure of ZX-999, according to the knowledge base?"

Expected:
  explicit refusal / not found
  no invented value

11) Conflict handling
To test conflict behavior later, add a second synthetic document containing a deliberately different design pressure for FV-201A.
Ask:
"What design pressure is authoritative for FV-201A?"

Expected:
  conflict is surfaced; system does not silently pick a value.

OBSERVABILITY TO CAPTURE
------------------------
For each test record:
  - user prompt
  - selected plan/capabilities
  - execution events/SSE
  - RAG candidates and source/page metadata
  - final answer
  - artifact id/hash when produced
  - sandbox attempt history for code tasks
