# LOCAL AI FOUNDATION — MASTER PROJECT CONTEXT AND CURRENT IMPLEMENTATION SPECIFICATION

## IMPORTANT: READ THIS ENTIRE DOCUMENT BEFORE MAKING ANY CODE CHANGES

This document defines the current state, architecture, scope, constraints, and implementation direction for the `Local AI Foundation` project.

The coding agent must:

1. Inspect the actual repository before assuming any file exists.
2. Treat the current verified inference infrastructure as working baseline infrastructure.
3. Avoid modifying protected runtime/setup areas unless explicitly necessary.
4. Work incrementally.
5. Explain the proposed implementation architecture and file changes before making substantial implementation decisions.
6. Keep the architecture modular and reusable.
7. Avoid prematurely implementing future systems such as full RAG, Docker infrastructure, PostgreSQL persistence, Qdrant, agents, or multimodal generation.
8. Do not turn this project into a chatbot application or an "AI wrapper."

The purpose of this project is much broader than chat.

---

# 1. THE BIG PICTURE

## 1.1 What this project is

This project is a modular, self-hosted local AI infrastructure foundation.

It is intended to become a reusable system that future applications can build on without those applications being tightly coupled to:

* one specific model,
* one model format,
* one inference runtime,
* one deployment topology,
* or one type of AI application.

The system is **not primarily a chatbot wrapper**.

The better mental model is:

> This is infrastructure for systems that happen to use AI models.

Future applications may be:

* chat systems,
* RAG applications,
* code analysis systems,
* coding assistants,
* agentic systems,
* document intelligence systems,
* data analysis systems,
* report generation systems,
* internal/private network AI services,
* tools that analyze files and produce structured results,
* workflows that generate PDFs or other artifacts,
* domain-specific applications,
* other software that needs local AI capabilities.

The AI Foundation should provide reusable capabilities.

The application above it decides how to combine those capabilities.

---

## 1.2 The architectural philosophy

The high-level architecture should eventually resemble:

```text
Applications / Consumers
        │
        ▼
Application-facing API / SDK
        │
        ▼
Local AI Foundation
        │
        ├── Model Management
        ├── Inference
        ├── Model Selection
        ├── Provider Management
        ├── Context Management
        ├── Structured Output Support
        ├── RAG / Knowledge Primitives
        ├── Embeddings
        ├── Retrieval
        ├── Reranking
        ├── Document / Data Processing Interfaces
        ├── Tool Interfaces
        └── Future Agent Primitives
        │
        ▼
Provider / Runtime Abstraction
        │
        ├── llama.cpp
        ├── future vLLM
        ├── future Transformers-based runtime
        ├── future embedding runtimes
        └── other appropriate runtimes
        │
        ▼
Models
        │
        ├── GGUF
        ├── SafeTensors
        ├── ONNX
        └── other future formats
        │
        ▼
GPU / CPU / Hardware
```

The fundamental principle is:

> Stable, reusable interfaces above. Replaceable, optimized execution infrastructure below.

Applications should not have to know the implementation details of the runtime whenever that can reasonably be avoided.

For example, an application should conceptually be able to request:

```text
Use model X.
Perform this inference.
Return the normalized result.
```

without having to directly care whether the execution path underneath is:

```text
llama.cpp + GGUF
```

or later:

```text
vLLM + SafeTensors
```

or another appropriate provider/runtime.

---

# 2. WHAT THE FOUNDATION IS EXPECTED TO EVENTUALLY SUPPORT

The project is intentionally broad in long-term scope.

However, not everything is being implemented now.

The long-term foundation should be able to support reusable primitives for the following categories.

---

## 2.1 Local inference

Support local models through one or more runtimes.

Current:

```text
llama.cpp
+
GGUF
```

Future possibilities include:

```text
vLLM
Transformers
specialized embedding runtimes
other optimized runtimes
```

The addition of another runtime should not require rewriting all consuming applications.

---

## 2.2 Model management

The system should eventually understand:

```text
known models
available models
missing models
model metadata
model aliases
model format
provider compatibility
capabilities
model roles
loaded model state
loading state
runtime errors
```

Examples of conceptual model roles may eventually include:

```text
general
coding
reasoning
embedding
reranking
vision
document
fast
quality
```

These are future metadata/capability concepts, not an instruction to hardcode all of them immediately.

---

## 2.3 Inference workloads

"Text" in this project does NOT mean merely chatbot conversations.

The LLM infrastructure may be used for:

```text
chat
question answering
code review
code generation
code analysis
agentic workflows
tool-assisted workflows
document analysis
data analysis
structured extraction
classification
summarization
report generation
JSON/structured output
RAG-based answering
long-context analysis
```

A workflow may ultimately produce:

```text
text
JSON
structured data
a report
a PDF
another application artifact
```

The LLM does not necessarily directly generate every final artifact.

For example:

```text
Data
   ↓
AI analysis
   ↓
Structured result
   ↓
Application/report generation layer
   ↓
PDF
```

The Foundation provides AI capabilities and reusable primitives. Higher layers may generate final artifacts.

---

## 2.4 RAG and knowledge systems

RAG is part of the long-term picture.

Eventually the Foundation may provide reusable primitives such as:

```text
document ingestion
document normalization
text extraction
chunking
embedding
indexing
vector storage
metadata storage
retrieval
reranking
context assembly
citation/source handling
```

A specific application should not necessarily be forced to use a fixed RAG implementation.

The Foundation should provide reusable building blocks.

For example:

```text
Application A:
    simple retrieval + inference

Application B:
    hybrid retrieval + reranking + inference

Application C:
    custom domain-specific retrieval pipeline
```

The lower reusable primitives should make these possible without forcing one rigid application workflow.

---

## 2.5 Documents, files, code, and data

Future systems using this Foundation may need to process:

```text
source code
repositories
documents
PDFs
structured data
logs
text files
tables
other application data
```

The Foundation should not be architected as if its only input is a single chat string.

However, this does NOT mean implementing all file/document ingestion now.

The architecture should simply avoid unnecessary assumptions that make these future workloads difficult.

---

## 2.6 Tools and agents

Eventually the system may support controlled interfaces for:

```text
tools
file access
retrieval
code execution
local scripts
other services
application-specific actions
```

Specific agent frameworks or applications may consume these capabilities.

The Foundation should provide reusable primitives.

It should not prematurely become a giant fixed agent framework.

---

## 2.7 Future non-text inputs

The current scope is not image or video generation.

However, the architecture should avoid unnecessarily hardcoding assumptions that make future multimodal support impossible.

Future possibilities may include:

```text
vision-language models
image understanding
document vision
OCR-assisted workflows
multimodal analysis
```

This is not immediate implementation scope.

The current project focus is primarily workloads involving language models, code, documents, data, structured information, and similar non-generation-of-image/video workflows.

---

# 3. LONG-TERM IMPLEMENTATION ROADMAP

The roadmap is conceptual and can evolve.

The current immediate stage is clearly identified later in this document.

---

## Stage 0 — Native infrastructure baseline

Establish one real, reproducible, verified local inference path.

This stage is substantially complete.

---

## Stage 1 — Core inference and model abstraction

Build the reusable Core around the already working runtime.

This is the CURRENT IMPLEMENTATION STAGE.

Core concepts include:

```text
model metadata
model registry
model availability
provider/runtime abstraction
provider manager
model lifecycle
normalized inference requests
normalized inference responses
safe state handling
```

Do not jump directly into RAG before this layer is stable.

---

## Stage 2 — Application-facing service/API layer

After the internal Core is stable, provide a clean way for applications to consume it.

Possible future forms:

```text
Python API
local HTTP API
SDK
service layer
```

The exact transport/API design should not contaminate the Core.

Core should remain reusable independently of one particular API transport.

---

## Stage 3 — Broader model/runtime support

Add additional providers where they genuinely provide value.

Examples may eventually include:

```text
vLLM
Transformers-based execution
embedding-specific runtimes
other optimized runtimes
```

The architecture should allow extension without requiring current implementation to support everything today.

---

## Stage 4 — Embeddings

Introduce dedicated embedding support.

Do not assume the general chat/generation model should be used for embeddings.

Conceptually support separate model categories:

```text
LLM
embedding model
reranker
```

---

## Stage 5 — RAG foundation

Implement reusable knowledge-system primitives:

```text
ingestion
normalization
chunking
embedding
indexing
storage
retrieval
context construction
```

At this stage, PostgreSQL and a vector database such as Qdrant may become relevant depending on the actual design.

Do not introduce them merely for the current model registry.

---

## Stage 6 — Better retrieval

Potentially add:

```text
hybrid retrieval
reranking
metadata filtering
retrieval strategies
```

---

## Stage 7 — Document intelligence

Potential future support:

```text
document extraction
PDF processing
scanned documents
OCR
layout handling
table extraction where useful
```

---

## Stage 8 — Context management

Reusable primitives may eventually manage:

```text
conversation history
summarization
context compression
RAG context
context windows
context limits
KV cache policies
```

This should be reusable across:

```text
chat
RAG
agents
tools
long documents
code analysis
```

---

## Stage 9 — Tools and agents

Only after the underlying primitives are stable.

Expose controlled capabilities.

Do not make the entire Foundation dependent on one agent framework.

---

## Stage 10 — Containerization and deployment

Docker should come later.

Potentially:

```text
Docker
Docker Compose
service-level deployment
```

Large model files should generally remain outside images and be mounted or otherwise supplied.

The already verified native path should remain understandable and reproducible.

---

# 4. CURRENT VERIFIED ENVIRONMENT AND FULL TECHNICAL BASELINE

## Project location

```text
~/Projects/local-ai
```

---

## Host operating system

```text
Ubuntu
```

---

## GPU

```text
NVIDIA GeForce RTX 5070
```

VRAM:

```text
12 GB
```

---

## NVIDIA driver

Verified:

```text
570.211.01
```

---

## CUDA

Verified CUDA Toolkit:

```text
12.8
```

The CUDA compiler intentionally used for this project:

```text
/usr/local/cuda-12.8/bin/nvcc
```

The system also contains an older Ubuntu-packaged CUDA compiler:

```text
/usr/bin/nvcc
```

Do not casually replace or alter this project configuration.

The project build explicitly used CUDA 12.8.

---

## Development tools

Verified:

```text
Git: 2.43.0
GCC: 13.3.0
G++: 13.3.0
Python: 3.12.3
CMake: 3.28.3
Ninja: 1.11.1
```

---

# 5. CURRENT INFERENCE RUNTIME

## Runtime

```text
llama.cpp
```

The source/runtime is located at:

```text
adapters/llama_cpp
```

It is tracked as an upstream Git submodule.

Verified revision:

```text
8887a48f050554f0ee59f56753860c061836b02d
```

Repository description at verification:

```text
b10736-7-g8887a48f0
```

This pinned runtime is part of the verified baseline.

Do not casually modify the submodule.

---

## Build system

```text
CMake
+
Ninja
```

Build type:

```text
Release
```

CUDA backend:

```text
enabled
```

Relevant verified build characteristics:

```text
CMAKE_BUILD_TYPE = Release
CMAKE_CUDA_COMPILER = /usr/local/cuda-12.8/bin/nvcc
CMAKE_GENERATOR = Ninja
GGML_CUDA = ON
GGML_CUDA_GRAPHS = ON
GGML_CUDA_FA = ON
```

---

# 6. CURRENT VERIFIED MODEL

Model location:

```text
models/gguf/Qwen3.5-9B-Q4_K_M.gguf
```

Model:

```text
Qwen3.5-9B
```

Format:

```text
GGUF
```

Quantization:

```text
Q4_K Medium
```

Approximate verified size:

```text
5.3 GB
```

Observed metadata included:

```text
Architecture: qwen35
Quantization: Q4_K Medium
Tensor count: 427
```

Model files are not source code and must not be committed to Git.

---

# 7. VERIFIED INFERENCE CONFIGURATION

The verified configuration includes:

```text
GPU layers: automatic
Context size: 4096
KV cache K: Q8_0
KV cache V: Q8_0
Batch size: 512
Micro-batch size: 256
Warmup: disabled
Reasoning: disabled
```

The equivalent relevant runtime behavior includes:

```text
--gpu-layers auto
--ctx-size 4096
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 512
--ubatch-size 256
--no-warmup
--reasoning off
```

Reasoning was explicitly disabled because the model/runtime was observed producing visible thinking-style output when reasoning was enabled/automatic.

The design principle is:

> Reasoning behavior should eventually be configurable through the Foundation rather than every application knowing runtime-specific flags.

---

# 8. VERIFIED PERFORMANCE

Observed during the earlier smoke test:

```text
Prompt processing: approximately 353 tokens/second
Generation: approximately 80 tokens/second
```

Observed during the local server API test, generation was approximately:

```text
around 90 tokens/second
```

These are machine/model/configuration-specific observations.

Do not treat them as universal benchmarks.

---

# 9. WHAT HAS ALREADY BEEN DONE

This project is NOT starting from zero.

The following work is already complete.

---

## 9.1 Native inference path established

The following real execution path was successfully verified:

```text
User Prompt
        ↓
llama.cpp
        ↓
CUDA backend
        ↓
RTX 5070
        ↓
GGUF model
        ↓
Generated response
```

This confirmed that the following work together:

```text
Ubuntu
NVIDIA driver
CUDA 12.8
llama.cpp
GGUF model
GPU inference
```

---

## 9.2 llama.cpp built successfully

The runtime was built with CUDA support.

The verified binary exists under:

```text
adapters/llama_cpp/build/bin/
```

Relevant executables include:

```text
llama-cli
llama-server
```

---

## 9.3 Smoke test implemented and verified

The repository contains:

```text
scripts/run_smoke_test.sh
```

The verified local model successfully produced:

```text
Local inference is working.
```

---

## 9.4 Reproducibility documentation created

The repository documents:

```text
README.md
docs/SETUP.md
docs/REPRODUCE.md
docs/MODELS.md
docs/INFERENCE.md
```

These document the verified environment and inference path.

---

## 9.5 Git repository initialized and committed

The repository currently has at least the following commits:

```text
f6e4f00 Initial reproducible local AI foundation
426378b Add reusable llama server launcher
```

The latest known state at the time of this specification is:

```text
426378b (HEAD -> master) Add reusable llama server launcher
```

The working tree was clean at the latest verification before beginning the next implementation phase.

---

# 10. LOCAL SERVER IMPLEMENTATION ALREADY COMPLETED

A reusable launcher exists:

```text
scripts/start_llama_server.sh
```

It validates:

```text
model file exists
llama-server binary exists and is executable
```

It then launches the server with the verified configuration.

The current launcher uses:

```text
HOST="127.0.0.1"
PORT="8080"
MODEL_ALIAS="qwen3.5-9b"
```

The model alias is:

```text
qwen3.5-9b
```

The current server command conceptually executes:

```text
llama-server
    --model <verified model>
    --alias qwen3.5-9b
    --host 127.0.0.1
    --port 8080
    --gpu-layers auto
    --ctx-size 4096
    --cache-type-k q8_0
    --cache-type-v q8_0
    --batch-size 512
    --ubatch-size 256
    --no-warmup
    --reasoning off
```

The server was successfully verified after startup.

---

# 11. VERIFIED LOCAL SERVER API

The local server listens on:

```text
127.0.0.1:8080
```

The models endpoint successfully returned the model alias:

```text
qwen3.5-9b
```

The following conceptual endpoint was successfully tested:

```text
/v1/models
```

A chat completion request was also successfully tested through:

```text
/v1/chat/completions
```

The model returned:

```text
The reusable local AI server is working.
```

The fact that the endpoint uses an OpenAI-compatible API shape does NOT mean this project uses OpenAI.

It simply means llama.cpp exposes a request/response format compatible with a widely adopted API convention.

The inference is entirely local:

```text
application/request
        ↓
local HTTP
        ↓
local llama-server
        ↓
local GGUF model
        ↓
local GPU
```

No OpenAI model or hosted inference is involved.

---

# 12. CURRENT REPOSITORY STRUCTURE

The latest inspected repository structure is approximately:

```text
local-ai/
│
├── core/
│   ├── config/
│   ├── inference/
│   └── models/
│
├── adapters/
│   └── llama_cpp/
│
├── api/
│
├── configs/
│
├── docs/
│   ├── INFERENCE.md
│   ├── MODELS.md
│   ├── REPRODUCE.md
│   └── SETUP.md
│
├── examples/
│
├── models/
│   └── gguf/
│       └── Qwen3.5-9B-Q4_K_M.gguf
│
├── scripts/
│   ├── build_llama_cpp.sh
│   ├── run_smoke_test.sh
│   └── start_llama_server.sh
│
├── tests/
│
├── .gitignore
├── .gitmodules
└── README.md
```

Important:

Some directories are currently intended architectural locations and may be empty.

The coding agent must inspect the actual repository rather than assuming all future files already exist.

---

# 13. STRICT PROTECTED AREAS — WHAT NOT TO MODIFY

## DO NOT MODIFY THESE UNLESS EXPLICITLY REQUIRED AND JUSTIFIED

```text
adapters/llama_cpp/
```

This is the external llama.cpp runtime submodule.

Do not:

* edit upstream llama.cpp source,
* casually update its revision,
* rebuild it unnecessarily,
* change its build configuration without justification,
* modify CUDA flags without a clear reason.

---

## DO NOT MODIFY THE VERIFIED MODEL

Do not:

```text
rename
move
delete
re-download
replace
commit
```

the verified model unless explicitly instructed.

Model files must not be committed to Git.

---

## DO NOT CASUALLY MODIFY THESE VERIFIED INFRASTRUCTURE FILES

```text
scripts/build_llama_cpp.sh
scripts/run_smoke_test.sh
scripts/start_llama_server.sh
```

They represent the verified baseline.

If a change is genuinely required, it must be clearly justified and should preserve or improve reproducibility.

---

## DO NOT PREMATURELY INTRODUCE

```text
Docker
Docker Compose
PostgreSQL
Qdrant
Redis
Kubernetes
a web frontend
a chatbot UI
a complete RAG pipeline
an agent framework
a vector database
authentication systems
multi-user infrastructure
remote deployment infrastructure
```

unless the current implementation task genuinely reaches that stage.

The current task is intentionally narrower.

---

# 14. STRICT WORKING DIRECTORIES FOR THE CURRENT IMPLEMENTATION

For the current Core implementation, work should primarily be limited to:

```text
core/
configs/
tests/
docs/
```

Potentially, if an intentionally small integration layer is required:

```text
api/
```

but do not begin building a large application-facing API unless explicitly part of the approved implementation plan.

Do not modify:

```text
adapters/llama_cpp/
models/
```

Do not modify existing runtime scripts unless explicitly necessary.

The coding agent should keep the current implementation isolated from the verified runtime infrastructure.

---

# 15. CURRENT ARCHITECTURAL DECISIONS

These decisions have already been discussed and should be respected.

---

## 15.1 Filesystem is not the complete model source of truth

The filesystem can determine:

```text
whether a model file exists
where it is located
possibly its format based on directory/extension
```

However, the filesystem alone cannot reliably provide all desired metadata such as:

```text
capabilities
intended provider
model role
runtime compatibility
other declared metadata
```

Therefore the conceptual model is:

```text
Filesystem
    → existence and path

Declared configuration / metadata
    → human-defined metadata and capabilities

Runtime/provider
    → actual load/runtime state
```

Do not pretend that scanning filenames magically discovers all model capabilities.

---

# 16. MODEL REGISTRY RESPONSIBILITIES

The Model Registry should conceptually own:

```text
known model definitions
configured model metadata
model identifiers
aliases where appropriate
expected paths
filesystem existence state
format/provider-related metadata
declared capabilities
```

The registry should NOT become the authoritative owner of:

```text
actual loaded runtime state
actual provider process state
actual inference health
```

The registry answers primarily:

> What models are known to this Foundation, and which of them currently appear available on disk?

---

## 16.1 Model availability is advisory

A registry may know that a model existed when it last refreshed.

The file may later:

```text
be deleted
be moved
become inaccessible
become corrupted
```

Therefore:

```text
registry availability
    !=
guaranteed runtime loadability
```

The provider/load operation remains the final authority.

The conceptual flow is:

```text
Registry says:
    model appears available

        ↓

Provider attempts actual load/use

        ↓

Success:
    model is usable

Failure:
    actual runtime error is authoritative
```

A failure may trigger an appropriate refresh/update.

Do not make stale registry information silently masquerade as guaranteed runtime truth.

---

# 17. PROVIDER / RUNTIME RESPONSIBILITIES

The Provider or Provider Manager should own runtime-facing state such as:

```text
which provider is active
which runtime handles a model
actual loaded state
loading state
runtime lifecycle
runtime errors
provider health
```

The provider is authoritative for:

> Is this model actually loaded and usable right now?

---

# 18. LOADED STATE OWNERSHIP

This is an important decision.

Do NOT have both:

```text
ModelRegistry
```

and:

```text
ProviderManager
```

independently claiming to own authoritative loaded state.

The ownership should be:

```text
ModelRegistry
    owns static/discovered model information

Provider / ProviderManager
    owns runtime-loaded state
```

If an API needs a combined view, the Core may compose:

```text
registry metadata
+
provider runtime state
```

but the provider remains authoritative for runtime state.

---

# 19. MODEL DISCOVERY / REFRESH PHILOSOPHY

Do not scan the model filesystem on every inference request.

Avoid:

```text
request
    ↓
scan all model directories
    ↓
infer everything again
```

Instead:

```text
startup / explicit refresh / meaningful trigger
    ↓
discover configured/available models
    ↓
maintain in-memory registry state
```

Possible future mechanisms may include:

```text
manual refresh
automatic refresh after model-management actions
filesystem watcher
event-driven updates
```

Do not implement filesystem watching unless it is part of the approved current implementation.

The current design should not make future event-driven updates impossible.

---

# 20. CONCURRENCY DECISION

The Foundation must NOT permanently force one concurrency model onto all future deployments.

Different applications and hardware may require:

```text
single-user
single-request
serialized
queued
multi-request
multi-user
network service
```

This policy should be adjustable above the lowest reusable Core layer where possible.

However:

> Lack of full concurrency support must not mean silent unsafe races.

Critical operations such as:

```text
load model
unload model
switch active model
provider lifecycle changes
```

should not silently race.

The current Core should at minimum be designed to:

```text
serialize unsafe conflicting operations
or
reject obviously conflicting operations clearly
```

The desired behavior is:

```text
fail loud and predictably
```

rather than:

```text
silent race
possibly inconsistent state
```

Do not prematurely build a complete distributed concurrency system.

The immediate goal is safe local state management.

---

# 21. CURRENT IMPLEMENTATION MILESTONE

The current milestone is:

> Build the first reusable Core layer above the already working llama.cpp runtime.

This should establish clean internal boundaries before future systems such as RAG are built.

The current implementation should focus on:

```text
model definitions
model metadata
model registry
availability checks
provider abstraction
provider management
runtime state ownership
model selection
normalized inference contracts
safe lifecycle/state handling
tests
documentation
```

---

# 22. SUGGESTED CORE ARCHITECTURE

The exact filenames may differ if a better clean architecture is justified.

However, responsibilities should remain approximately equivalent.

Possible conceptual structure:

```text
core/
│
├── config/
│   ├── settings
│   └── configuration loading
│
├── models/
│   ├── model definitions
│   ├── metadata structures
│   ├── registry
│   └── discovery/availability
│
├── inference/
│   ├── request definitions
│   ├── response definitions
│   ├── provider interface
│   ├── provider manager
│   ├── lifecycle/state
│   └── inference coordination
│
└── common/
    ├── errors
    └── shared primitives
```

Do not create unnecessary abstractions merely for theoretical elegance.

The first implementation should be small, understandable, and robust.

---

# 23. MODEL DEFINITION REQUIREMENTS

A model definition should be able to represent concepts such as:

```text
stable internal identifier
display name
path
format
provider/runtime compatibility
declared capabilities
availability
other useful metadata
```

Do not hardcode the current Qwen model throughout the Core.

The current model should eventually be represented as configuration/data rather than as a special case in business logic.

The exact schema should remain reasonably extensible.

Do not over-design for hundreds of fields that are not currently needed.

---

# 24. CONFIGURATION REQUIREMENTS

The current system should move toward configuration-driven model definitions where appropriate.

A model addition should eventually conceptually resemble:

```text
1. Download model.
2. Place model in the expected model storage location.
3. Add or update declared metadata/configuration if necessary.
4. Refresh/reload the registry.
5. The model becomes visible to the Foundation.
```

The system should minimize unnecessary manual work.

However, do not falsely claim automatic discovery can determine metadata that is not actually encoded reliably in the model.

Existence/discovery can be automated.

Capabilities and intentional metadata may require configuration.

---

# 25. PROVIDER ABSTRACTION REQUIREMENTS

The public Core should not directly scatter llama.cpp-specific assumptions throughout the entire codebase.

The provider abstraction should make it possible to eventually support:

```text
llama.cpp
vLLM
other runtimes
```

without rewriting every Core consumer.

However, do NOT build fake implementations for providers that do not exist yet.

The immediate implementation should support the real current provider path.

Future providers should be possible through the interface.

Do not implement speculative vLLM integration now.

---

# 26. NORMALIZED INFERENCE CONTRACTS

The Core should move toward normalized internal request/response structures.

Applications should not need to depend directly on the exact JSON structure returned by llama.cpp.

Conceptually:

```text
Application/Core Consumer
        ↓
Normalized Inference Request
        ↓
Provider
        ↓
Runtime-specific request
        ↓
Runtime
        ↓
Runtime-specific response
        ↓
Provider normalization
        ↓
Normalized Inference Response
```

The current provider may internally communicate with:

```text
llama-server
```

using its local API.

Do not expose llama.cpp response structures as the permanent Core abstraction unless deliberately required.

---

# 27. CURRENT LLAMA.CPP INTEGRATION BOUNDARY

The current verified runtime is launched externally using:

```text
scripts/start_llama_server.sh
```

It exposes:

```text
http://127.0.0.1:8080
```

The current model alias is:

```text
qwen3.5-9b
```

The Core implementation may communicate with the running local runtime through its API.

Do not automatically redesign the verified launcher unless there is a strong architectural reason.

The Core should not require editing llama.cpp itself.

---

# 28. CURRENT MODEL LOADING PHILOSOPHY

There is an important distinction between:

```text
known
available
loaded
usable
```

These should not be treated as identical.

Conceptually:

```text
KNOWN
    Registry knows about model definition.

AVAILABLE
    Expected model file currently exists/is accessible.

LOADING
    Provider is currently attempting to load or activate it.

LOADED
    Provider reports it as loaded/active.

USABLE
    Provider/runtime can actually successfully serve requests.
```

The exact state model may be simplified initially if necessary.

But do not collapse everything into one boolean like:

```text
is_model_good
```

The architecture should preserve these conceptual distinctions.

---

# 29. ERROR HANDLING REQUIREMENTS

The system should distinguish categories such as:

```text
unknown model
configured model missing from filesystem
unsupported format
unsupported provider
provider unavailable
model load failure
provider communication failure
invalid inference request
runtime failure
unsafe concurrent lifecycle operation
```

Errors should be clear enough for a future API/application layer to present meaningful information.

Avoid leaking random low-level implementation exceptions everywhere without context.

At the same time, do not build a giant enterprise error framework.

Keep it simple and explicit.

---

# 30. TESTING REQUIREMENTS

The current implementation should include tests for Core behavior that does not require loading the 5.3 GB model every time.

Examples include:

```text
model registry behavior
configuration parsing
availability checks
unknown model behavior
missing file behavior
provider selection
state ownership
conflicting lifecycle operations
normalized request/response behavior
```

Do not make every unit test dependent on:

```text
GPU
CUDA
real model loading
live llama-server
```

Integration tests may exist separately for real runtime verification.

---

# 31. DOCUMENTATION REQUIREMENTS

Update or add documentation for the newly introduced Core architecture.

Documentation should clearly explain:

```text
what Core owns
what the registry owns
what providers own
what runtime state means
how model configuration works
how model refresh works
how applications eventually consume the system
```

Do not overwrite the existing verified environment documentation with speculative information.

Preserve the distinction between:

```text
verified current facts
```

and:

```text
future architecture
```

---

# 32. DO NOT IMPLEMENT RAG YET

RAG is absolutely part of the future system.

However, do not implement:

```text
PostgreSQL
Qdrant
document ingestion
chunking
embedding pipeline
vector indexing
retrieval pipeline
```

during the current milestone.

The reason is architectural sequencing.

We first need:

```text
stable model representation
+
stable inference/provider abstraction
+
stable lifecycle behavior
```

Then RAG can consume those capabilities.

---

# 33. DOCKER STATUS

Docker is intentionally not part of the current milestone.

The native execution path is already verified:

```text
Ubuntu
    ↓
NVIDIA driver
    ↓
CUDA
    ↓
llama.cpp
    ↓
GGUF model
    ↓
GPU
```

This path should remain reproducible and understandable.

Containerization may later wrap service-level components.

Models should generally remain outside container images.

---

# 34. POSTGRESQL STATUS

PostgreSQL is not currently required for:

```text
model registry
model discovery
model availability
provider state
```

Do not introduce PostgreSQL merely to maintain a list of model files.

PostgreSQL may become relevant later for persistent application or knowledge metadata.

---

# 35. VECTOR DATABASE STATUS

Qdrant or another vector database may become relevant when the RAG/knowledge system is implemented.

It is not required for the current Core model/inference layer.

---

# 36. FUTURE PRIVATE NETWORK HOSTING

The architecture should remain compatible with future private network deployment.

For example, a future deployment may expose the Foundation to trusted applications on a private network.

However, do not implement that network service/deployment model now.

The current verified server intentionally binds to:

```text
127.0.0.1
```

Do not casually expose the inference runtime to the LAN or internet.

Security and deployment policy belong to a later deployment stage.

---

# 37. MODEL FORMAT EXTENSIBILITY

The model directory is conceptually organized by format:

```text
models/
├── gguf/
├── safetensors/
├── onnx/
└── other/
```

The model format should not automatically define the public interface.

Instead:

```text
application
    ↓
Foundation
    ↓
provider/runtime selection
    ↓
appropriate runtime for format/model
```

Current actual support is only the verified llama.cpp + GGUF path.

Do not claim SafeTensors or ONNX support is implemented.

The architecture should merely allow future addition.

---

# 38. IMPORTANT NON-GOAL: DO NOT MAKE THIS AN AI WRAPPER

Avoid creating an architecture like:

```text
send_prompt()
    ↓
call llama.cpp
    ↓
return text
```

and calling that the entire Foundation.

The project should be capable of supporting broader workflows.

At the same time, avoid overengineering the current Core into a giant platform.

The correct balance is:

> Build small, strong, reusable primitives that future systems can compose.

---

# 39. REQUIRED CODING AGENT WORKFLOW

Before creating significant implementation files:

## Step 1 — Inspect

Inspect:

```text
repository tree
existing source files
current empty/non-empty directories
current configuration files
current tests
Git status
```

Do not assume this specification perfectly reflects every current file.

The repository itself is authoritative for current implementation state.

---

## Step 2 — Report implementation plan

Before making significant code changes, provide:

```text
proposed directories
proposed files
responsibility of each file
data flow
state ownership
model registry design
provider interface design
error handling approach
testing approach
```

Keep the plan specific to the current milestone.

---

## Step 3 — Check against protected boundaries

Explicitly confirm that the plan does NOT unnecessarily modify:

```text
adapters/llama_cpp/
models/
verified CUDA setup
verified runtime scripts
```

---

## Step 4 — Implement incrementally

Implement the Core in small logical units.

Avoid dumping a huge speculative framework into the repository.

---

## Step 5 — Test

Run unit tests where applicable.

Do not require the real model/GPU for all tests.

---

## Step 6 — Review

Show:

```text
files created
files modified
important architectural decisions
tests run
test results
anything intentionally deferred
```

---

# 40. CURRENT IMPLEMENTATION PRIORITY ORDER

The current recommended order is:

## Milestone A — Core data contracts

Define clean structures for:

```text
model metadata
model identifiers
availability information
capabilities
inference request
inference response
errors
```

---

## Milestone B — Model Registry

Implement:

```text
model definitions
configuration loading
registry initialization
availability checks
listing known models
lookup by identifier
refresh behavior
missing model handling
```

Do not scan everything for every request.

---

## Milestone C — Provider abstraction

Define the provider interface.

Support the real current llama.cpp provider path.

Do not implement fake future providers.

---

## Milestone D — Provider/runtime management

Implement ownership of:

```text
provider state
runtime communication
loaded/loading state
safe lifecycle operations
provider errors
```

The provider is authoritative for runtime state.

---

## Milestone E — Normalized inference

Connect:

```text
Core request
    ↓
provider
    ↓
llama.cpp server
    ↓
normalized Core response
```

Keep llama.cpp-specific details contained within the provider boundary.

---

## Milestone F — Tests and documentation

Verify:

```text
registry behavior
missing model behavior
provider selection
request normalization
response normalization
error behavior
unsafe lifecycle conflicts
```

Document the architecture.

---

# 41. DEFINITION OF SUCCESS FOR THE CURRENT MILESTONE

The current milestone is successful when the system can conceptually do something like:

```text
Core knows model definitions
        ↓
Core can determine which configured models appear available
        ↓
Caller selects model by stable identifier
        ↓
Core selects the appropriate provider
        ↓
Provider communicates with the runtime
        ↓
Inference request is normalized
        ↓
Runtime executes inference
        ↓
Provider normalizes the result
        ↓
Caller receives a stable Foundation-level response
```

while preserving these boundaries:

```text
Registry
    ≠
runtime authority

Provider
    =
runtime authority

Filesystem
    =
model existence/path authority

Configuration
    =
declared metadata/capability authority
```

---

# 42. WHAT SHOULD REMAIN OUT OF SCOPE FOR THIS MILESTONE

Strictly defer:

```text
full RAG
PostgreSQL
Qdrant
Docker
Docker Compose
web UI
chatbot application
agent framework
tool execution framework
OCR
vision inference
image generation
video generation
multi-user service architecture
authentication
internet exposure
Kubernetes
distributed inference
advanced scheduling
full filesystem watcher/event bus
vLLM implementation
SafeTensors runtime implementation
ONNX runtime implementation
```

The architecture may leave extension points for some of these.

Do not implement them.

---

# 43. FINAL PROJECT PRINCIPLE

The goal is not:

> Build a single application that happens to call an LLM.

The goal is:

> Build a stable local AI infrastructure foundation whose reusable capabilities can support many different applications.

The system should gradually become capable of supporting:

```text
models
        ↓
providers/runtimes
        ↓
inference
        ↓
knowledge/RAG primitives
        ↓
documents/data/code workflows
        ↓
tools/agents
        ↓
applications
```

without forcing every future application into one fixed workflow.

The immediate work is only the next correct layer:

> Build the reusable Core model/inference/provider foundation above the already verified llama.cpp runtime.

Do not skip architectural layers merely because RAG, agents, PostgreSQL, Docker, or future runtimes are eventually desired.

Build the foundation deliberately, keep the verified native runtime intact, and create clean boundaries so future capabilities can be added without requiring a rewrite.
