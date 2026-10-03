from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .domain import MomState, StageName
from .models import Meeting, Provenance, Segment


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL); INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_version); CREATE TABLE IF NOT EXISTS meetings(id TEXT PRIMARY KEY,title TEXT,date TEXT,topic TEXT,folder TEXT,mom_state TEXT DEFAULT 'Draft'); CREATE TABLE IF NOT EXISTS recordings(meeting_id TEXT,path TEXT,ord INTEGER,duration REAL); CREATE TABLE IF NOT EXISTS transcript_segments(meeting_id TEXT,id TEXT,start REAL,end REAL,text TEXT,signals TEXT,flags TEXT,speaker TEXT DEFAULT 'Speaker 1',recording_index INTEGER DEFAULT 0,recording_start REAL,recording_end REAL,PRIMARY KEY(meeting_id,id)); CREATE TABLE IF NOT EXISTS english_segments(meeting_id TEXT,id TEXT,text TEXT,PRIMARY KEY(meeting_id,id)); CREATE TABLE IF NOT EXISTS stages(meeting_id TEXT,name TEXT,complete INTEGER,PRIMARY KEY(meeting_id,name)); CREATE TABLE IF NOT EXISTS artefacts(meeting_id TEXT,name TEXT,revision INTEGER,path TEXT,provenance TEXT,content TEXT,PRIMARY KEY(meeting_id,name)); CREATE TABLE IF NOT EXISTS extracted_facts(meeting_id TEXT,chunk_index INTEGER,facts TEXT,PRIMARY KEY(meeting_id,chunk_index));"""
        )
        self.db.commit()

    def create_meeting(self, meeting: Meeting) -> None:
        self.db.execute(
            "INSERT INTO meetings(id,title,date,topic,folder,mom_state) VALUES(?,?,?,?,?,?)",
            (
                meeting.id,
                meeting.title,
                meeting.date.isoformat(),
                meeting.topic,
                str(meeting.folder),
                MomState.DRAFT.value,
            ),
        )
        self.db.commit()

    def add_recording(
        self, meeting_id: str, path: Path, order: int, duration: float
    ) -> None:
        self.db.execute(
            "INSERT INTO recordings VALUES(?,?,?,?)",
            (meeting_id, str(path), order, duration),
        )
        self.db.commit()

    def recordings(self, meeting_id: str) -> list[tuple[Path, float]]:
        return [
            (Path(r["path"]), float(r["duration"]))
            for r in self.db.execute(
                "SELECT path,duration FROM recordings WHERE meeting_id=? ORDER BY ord",
                (meeting_id,),
            )
        ]

    def is_stage_complete(self, meeting_id: str, stage: StageName) -> bool:
        row = self.db.execute(
            "SELECT complete FROM stages WHERE meeting_id=? AND name=?",
            (meeting_id, stage.value),
        ).fetchone()
        return bool(row and row[0])

    def mark_stage_complete(self, meeting_id: str, stage: StageName) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO stages VALUES(?,?,1)", (meeting_id, stage.value)
        )
        self.db.commit()

    def save_segments(self, meeting_id: str, segments: list[Segment]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO transcript_segments(meeting_id,id,start,end,text,signals,flags,speaker,recording_index,recording_start,recording_end) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    meeting_id,
                    s.id,
                    s.start,
                    s.end,
                    s.text,
                    json.dumps(
                        {
                            "avg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                            "compression_ratio": s.compression_ratio,
                            "word_probabilities": s.word_probabilities,
                            "confidence_signals": s.confidence_signals,
                        }
                    ),
                    json.dumps(s.flags),
                    s.speaker,
                    s.recording_index,
                    s.recording_start if s.recording_start is not None else s.start,
                    s.recording_end if s.recording_end is not None else s.end,
                )
                for s in segments
            ],
        )
        self.db.commit()

    def segments(self, meeting_id: str) -> list[Segment]:
        def signals(row: sqlite3.Row) -> dict[str, Any]:
            values = cast(dict[str, Any], json.loads(row["signals"]))
            values["word_probabilities"] = tuple(values.get("word_probabilities", ()))
            return values

        return [
            Segment(
                r["id"],
                r["start"],
                r["end"],
                r["text"],
                **signals(r),
                flags=json.loads(r["flags"]),
                speaker=r["speaker"],
                recording_index=r["recording_index"],
                recording_start=r["recording_start"],
                recording_end=r["recording_end"],
            )
            for r in self.db.execute(
                "SELECT * FROM transcript_segments WHERE meeting_id=? ORDER BY start,id",
                (meeting_id,),
            )
        ]

    def save_english(self, meeting_id: str, english: dict[str, str]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO english_segments VALUES(?,?,?)",
            [(meeting_id, key, value) for key, value in english.items()],
        )
        self.db.commit()

    def english(self, meeting_id: str) -> dict[str, str]:
        return dict(
            self.db.execute(
                "SELECT id,text FROM english_segments WHERE meeting_id=?", (meeting_id,)
            ).fetchall()
        )

    def save_facts(self, meeting_id: str, index: int, facts: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO extracted_facts VALUES(?,?,?)",
            (meeting_id, index, json.dumps(facts)),
        )
        self.db.commit()

    def facts(self, meeting_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT facts FROM extracted_facts WHERE meeting_id=? ORDER BY chunk_index",
                (meeting_id,),
            )
        ]

    def artefact(
        self,
        meeting_id: str,
        name: str,
        path: Path,
        provenance: Provenance,
        content: str,
    ) -> int:
        row = self.db.execute(
            "SELECT revision FROM artefacts WHERE meeting_id=? AND name=?",
            (meeting_id, name),
        ).fetchone()
        revision = int(row[0]) + 1 if row else 1
        self.db.execute(
            "INSERT OR REPLACE INTO artefacts VALUES(?,?,?,?,?,?)",
            (
                meeting_id,
                name,
                revision,
                str(path),
                json.dumps(asdict(provenance)),
                content,
            ),
        )
        self.db.commit()
        return revision

    def artefact_revision(self, meeting_id: str, name: str) -> int:
        row = self.db.execute(
            "SELECT revision FROM artefacts WHERE meeting_id=? AND name=?",
            (meeting_id, name),
        ).fetchone()
        return int(row[0]) if row else 0

    def artefact_content(self, meeting_id: str, name: str) -> str | None:
        row = self.db.execute(
            "SELECT content FROM artefacts WHERE meeting_id=? AND name=?",
            (meeting_id, name),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def provenance(self, meeting_id: str) -> dict[str, Provenance]:
        return {
            row["name"]: Provenance(**json.loads(row["provenance"]))
            for row in self.db.execute(
                "SELECT name,provenance FROM artefacts WHERE meeting_id=?",
                (meeting_id,),
            )
        }

    def close(self) -> None:
        self.db.close()
