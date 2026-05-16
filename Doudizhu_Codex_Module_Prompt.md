# Codex Module Implementation Prompt

## Role and Objective

You are an expert Quantitative System Developer and Senior Software Architect. Your task is to implement one assigned module for a Doudizhu AI Assistant based on the PerfectDou reinforcement learning architecture.

## Architectural Pipeline Context

The system uses a strict single-thread asynchronous pipeline:

`IO_READ -> CV_INFERENCE -> STATE_NORMALIZATION -> RL_FORWARD -> UI_RENDER`

You will receive one specific file, class, or method to implement. You must stay inside the assigned module boundary. Do not move logic into another pipeline step and do not create hidden cross-step state.

## Module Assignment

Fill this section before sending the prompt to Codex:

```text
ASSIGNED_FILE:
ASSIGNED_CLASS_OR_FUNCTION:
PIPELINE_STEP:
LANGUAGE:
INPUT_CONTRACT:
OUTPUT_CONTRACT:
ALLOWED_DEPENDENCIES:
FORBIDDEN_DEPENDENCIES:
TESTS_TO_CREATE_OR_UPDATE:
PERFORMANCE_TARGET:
```

## Strict Coding Guidelines

1. Object-oriented integrity and explicit referencing
   - When accessing or mutating member variables or methods inside a class, always use `this.`.
   - If the language is Python, always use `self.` in the same spirit.

2. Single-exit loop discipline
   - Do not use `break` or `continue`.
   - Control loop termination with explicit boolean flags or loop conditions.
   - Keep each method organized around one logical return path wherever practical.

3. Defensive programming with context
   - Use strict type definitions and safe initialization as the primary safety mechanism.
   - Add null checks only where the input contract genuinely permits null or external data makes null possible.
   - Do not add noisy defensive code that hides contract violations.

4. State normalization
   - In `state_manager`, persist only base truths: initial deal, landlord role, current turn pointer, and atomic action logs.
   - Do not persist derived state such as remaining cards, inferred opponent hands, legal action cache, or tensor matrices.
   - Derive remaining cards, current state matrix, and legal views from base truths on demand.

5. Pipeline boundary discipline
   - `IO_READ` captures frames only.
   - `CV_INFERENCE` converts frames into observed card detections only.
   - `STATE_NORMALIZATION` cleans observations and maintains normalized base state only.
   - `RL_FORWARD` builds tensors, runs model inference, maps probabilities to legal actions, and returns recommendation data only.
   - `UI_RENDER` renders recommendations only and must not mutate game state or invoke model logic.

## Required Output Format

Output only:

1. A complete code block for the requested module.
2. A brief plain-text explanation of the internal execution logic.

Do not include unrelated files, broad architecture commentary, or speculative future work unless the assignment explicitly asks for it.

## Quality Gate

Before finalizing the module, verify:

- The implementation stays within the assigned file and pipeline step.
- All class member references use `this.` or `self.`.
- No `break` or `continue` appears.
- State normalization rules are respected.
- Inputs and outputs match the assignment contract.
- Edge cases are handled by contract-aware validation rather than blanket null checks.
