# EA-AI-Assistant

An AI assistant for Enterprise Architecture work. It is built for one user first and is designed so that other individuals can each have their own later.

## Language

**Enterprise Architecture (EA)**:
The discipline of describing and steering how an organisation's business capabilities, information, applications and technology fit together. In this project "EA" always means this, never "Executive Assistant".
_Avoid_: Executive Assistant (for EA)

### Meetings

**Meeting**:
A single discussion session the user attended and wants documented. A Meeting has one or more Recordings, taken in a user-given order, and belongs to at most one Topic.
_Avoid_: Session, call

**Topic**:
A user-defined subject (typically a solution or initiative under design) that groups related Meetings so their content can be used together, e.g. to produce an HLD.
_Avoid_: Project, initiative, group, folder

**Recording**:
One audio file captured from a Meeting, as supplied by the user. It is raw input and may be discarded once the Meeting's MoM is approved.
_Avoid_: Record, audio file, media

**Transcript**:
The verbatim text of everything said in a Meeting (all its Recordings joined in order), in the language(s) actually spoken (Egyptian Arabic mixed with English), attributed to Speakers. It is the source of truth for everything generated from the Meeting.
_Avoid_: Transcription (for the artefact), subtitles, text

**English Transcript**:
A faithful English translation of a Transcript, preserving Speakers and order. It is what the Summary and MoM are written from.
_Avoid_: Translation, translated transcript

**Speaker**:
A distinct voice in a Transcript. Starts anonymous ("Speaker 1") and may be named by the user.
_Avoid_: Participant (for the voice), attendee

**Flagged Passage**:
A part of a Transcript the system has detected as an accuracy risk, with a stated reason, and asks the user to check. Unflagged text is not guaranteed correct. Correcting one makes the Meeting's outputs Stale.
_Avoid_: Warning, error, low-confidence segment

**Vocabulary**:
The user-maintained list of proper names, system names and acronyms, each with one canonical spelling and any aliases it may be heard as (Arabic script, English, acronym). There is one global Vocabulary and optionally one per Topic.
_Avoid_: Dictionary, glossary (reserved for this file), terms list

### Meeting outputs

**Summary**:
A short English narrative of what was discussed in a Meeting.
_Avoid_: Recap, overview, abstract

**Minutes of Meeting (MoM)**:
A structured English record of a Meeting: purpose, discussion points, numbered Decisions, Action Items, open questions, next steps and Technical Details. It is Draft until the user approves it.
_Avoid_: Minutes, notes, meeting report

**Approved MoM**:
A specific version of a MoM the user has confirmed as correct. It stays the Meeting's Approved MoM until the user approves a newer version. Only Approved MoMs are used beyond their own Meeting (e.g. for an HLD).
_Avoid_: Final MoM, signed-off minutes

**Stale**:
The state of an English Transcript, Summary or MoM whose inputs (Transcript, English Transcript, Speaker names, Recording order) have changed since it was generated.
_Avoid_: Outdated, dirty

**Evidence**:
The Transcript passages a Decision, Action Item or Technical Details entry was derived from, so the user can check it against what was actually said.
_Avoid_: Source, citation, reference

**Decision**:
A conclusion that the Meeting agreed on and that a MoM records.
_Avoid_: Resolution, outcome

**Action Item**:
A task agreed in a Meeting, with an owner and a due date when these were stated.
_Avoid_: Task, to-do, follow-up

**Technical Details**:
The section of a MoM that captures architecture-relevant facts mentioned in the Meeting: systems, integrations, data flows, volumes, non-functional requirements and constraints.
_Avoid_: Design notes, tech notes

### Topic outputs

**Attachment**:
A document the user adds to a Topic (e.g. requirements, existing designs, Mermaid diagrams) to be used alongside its Approved MoMs.
_Avoid_: Upload, reference doc, supporting material

**High-Level Design (HLD)**:
An architecture document for a solution, produced on the user's request from the Approved MoMs and Attachments of one Topic, following the user's HLD template.
_Avoid_: Design doc, architecture doc, solution design
