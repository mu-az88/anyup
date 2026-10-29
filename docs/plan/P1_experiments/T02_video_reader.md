# T02 — Video Reader + Chunked Backbone Runner

| Field | Value |
|---|---|
| Task ID | T02 |
| Plan | P1 Experiments (`plan.md`), Phase A.1/A.3 |
| Status | Q&A (partial; more questions to ask before implementing) |
| Depends on | T01 (backbone adapters) |
| Blocks | T03 Upsampler wrapper, notebook refactor |

## Goal

A helper that takes any video (any resolution, any frame rate) and runs a backbone on **every** frame, with no frame skipping. Long videos are split into chunks that are loaded and processed one at a time, so memory stays constant.

## Scope

- In scope: reading a video into [0,1] frame chunks, splitting it into chunks per backbone, handling the last chunk, and yielding `BackboneOutput` per chunk with global frame indices.
- Out of scope: the backbone adapters themselves (T01), upsampling (T03).

## Q&A (decisions agreed before implementation)

| # | Question | Answer |
|---|---|---|
| 1 | Own task or part of T01? | **Own task (T02).** T01 adapters take one chunk `(T, 3, H, W)`; T02 handles reading and chunking. |
| 2 | Frame use relative to the old Exp 2 (16 linspace) / Exp 3 (every 12th) protocols? | **Always all frames.** No skipping anywhere. Exp 2/3 are redefined on all frames, so the thesis numbers can't be reproduced exactly (logged in `plan.md`). |
| 3 | VideoMAE last chunk when the length isn't a multiple of 16? | **Pad with the last frame** up to 16, then drop the outputs of the padded frames. |
| 4 | Return type for long videos? | **Iterator of chunks:** yields one `BackboneOutput` per chunk, with global frame indices in `meta`. |

_Still to ask when T02 starts:_ input sources (video files / frame folders / tensors) and the decoding library; aspect ratio vs fixed square input; chunk size and overlap for per-frame backbones; how padded/dropped frames show up in `meta`; how short videos (< 1 chunk) are handled; how this interacts with AnyUp3D's temporal context across chunk boundaries (T03).

## Design

_To be written after the Q&A is complete._

## Steps

_To be written._

## Tests / Regression Checks

_To be written._

## Done When

_To be written._

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._
