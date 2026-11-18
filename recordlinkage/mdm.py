"""Multi-source customer master data with incremental merging.

This module implements a master data management (MDM) layer on top of the
record linkage pipeline (:class:`recordlinkage.Index`,
:class:`recordlinkage.Compare` and a fitted probabilistic classifier).

Records are identified by a composite key ``(source, record_id)``. A batch
of incremental operations (``I`` insert, ``U`` update, ``D`` delete) is
linked against the current master state with the *same frozen* Index,
Compare and probability model objects. Candidate pairs are scored with the
fitted model and an edge (connection) is accepted when the match
probability reaches a configurable threshold. Connected components of the
edge graph form the current entities; records without any connection
(isolated records) become singleton entities.

The complete state (raw records, connections, entities, members and fact
references) is persisted in SQLite. Every batch is published atomically as
a new *generation*: if scoring, a constraint check, a ``pre_commit`` hook
or the database commit fails, the previous generation remains the current
state and reopening the database simply continues from it.

Entity identifiers are stable across generations:

* a **merge** keeps the smallest identifier of the merged (old anchor)
  entities;
* a **split** lets the subgroup that still contains the old anchor record
  inherit the identifier, while every other subgroup takes the smallest
  member key;
* a brand new component (no old anchor) takes the smallest member key.

Facts registered against source records keep their payload untouched.
Their reference to the *current* entity is rebound after every batch,
including when the anchor of an entity moves due to a merge or split.
Deleting the referenced source record sets the fact reference to
``NULL``; the fact itself is never removed.
"""

import json
import sqlite3

import numpy
import pandas

from recordlinkage.base import BaseCompare
from recordlinkage.base import BaseIndex

#: Operations accepted in an incremental batch.
VALID_OPERATIONS = ("I", "U", "D")

_META_GENERATION = "current_generation"


class MDMError(Exception):
    """Base class for errors raised by the MDM layer."""


class MDMOperationError(MDMError):
    """Raised when an incremental batch is malformed or inconsistent."""


class MDMPublishError(MDMError):
    """Raised when a batch could not be published atomically."""


def _key_to_json(source, record_id):
    """Return the canonical, sortable string encoding of a record key."""

    return json.dumps([source, record_id], ensure_ascii=False)


def _json_default(obj):
    """JSON fallback for numpy/pandas scalar values."""

    if isinstance(obj, numpy.generic):
        return obj.item()
    if isinstance(obj, pandas.Timestamp):
        return obj.isoformat()
    if obj is None or (isinstance(obj, float) and numpy.isnan(obj)):
        return None
    if pandas.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, default=_json_default, allow_nan=False)


def _loads(value):
    if value is None:
        return None
    return json.loads(value)


def _scalar(value):
    """Coerce a key column value to a JSON serializable scalar."""

    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, float) and numpy.isnan(value):
        raise MDMOperationError("source and record_id may not be missing")
    if pandas.isna(value):
        raise MDMOperationError("source and record_id may not be missing")
    if not isinstance(value, (str, int, float, bool)):
        raise MDMOperationError(
            f"source and record_id must be scalar values, got {type(value).__name__}"
        )
    return value


def _sanitize_mapping(mapping):
    """Return a JSON clean copy of a payload mapping (NaN -> None)."""

    clean = {}
    for column, value in dict(mapping).items():
        if value is None:
            clean[column] = None
        elif isinstance(value, float) and numpy.isnan(value):
            clean[column] = None
        elif pandas.isna(value) and not isinstance(value, (str, bytes)):
            clean[column] = None
        elif isinstance(value, numpy.generic):
            clean[column] = value.item()
        else:
            clean[column] = value
    # validate that the whole payload is JSON serializable
    _dumps(clean)
    return clean


class _UnionFind:
    """Minimal union-find structure used to build connected components."""

    def __init__(self, elements):
        self._parent = {element: element for element in elements}

    def find(self, element):
        parent = self._parent[element]
        if parent != element:
            parent = self.find(parent)
            self._parent[element] = parent
        return parent

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self._parent[root_right] = root_left

    def components(self):
        groups = {}
        for element in self._parent:
            groups.setdefault(self.find(element), []).append(element)
        return list(groups.values())


class MDMManager:
    """Incremental, SQLite backed multi-source master data manager.

    Parameters
    ----------
    path : str or path-like
        SQLite database file. The file is created when it does not exist.
        Reopening an existing database continues from the latest published
        generation.
    indexer : recordlinkage.base.BaseIndex
        A configured indexer, e.g. ``recordlinkage.Index().block(...)``.
        The object is used as-is and never re-fitted.
    compare : recordlinkage.base.BaseCompare
        A configured compare object with the comparison features. The
        object is used as-is and never re-fitted.
    classifier : recordlinkage.base.BaseClassifier
        An already fitted probabilistic classifier exposing a working
        ``prob(comparison_vectors)`` method. The classifier is treated as
        frozen: only ``prob`` is called, ``fit`` is never invoked.
    threshold : float, optional
        Minimum match probability (inclusive) for a candidate pair to
        become a connection. Default ``0.5``.
    pre_commit : callable, optional
        Optional hook ``pre_commit(connection)`` invoked inside the batch
        transaction, just before the commit. Raising from the hook aborts
        the batch and keeps the previous generation.

    Attributes
    ----------
    threshold : float
        The configured match probability threshold.
    generation : int
        Generation number of the current (latest published) state.
    """

    def __init__(
        self, path, indexer, compare, classifier, threshold=0.5, pre_commit=None
    ):
        if not isinstance(indexer, BaseIndex):
            raise TypeError("indexer must be a recordlinkage Index instance")
        if not isinstance(compare, BaseCompare):
            raise TypeError("compare must be a recordlinkage Compare instance")
        if not callable(getattr(classifier, "prob", None)):
            raise TypeError(
                "classifier must be a fitted classifier with a prob(...) method"
            )
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("threshold must be a probability between 0 and 1")
        if pre_commit is not None and not callable(pre_commit):
            raise TypeError("pre_commit must be callable")

        self.path = str(path)
        self.indexer = indexer
        self.compare = compare
        self.classifier = classifier
        self.threshold = float(threshold)
        self.pre_commit = pre_commit

        # autocommit mode so that batch transactions are fully controlled
        # with explicit BEGIN/COMMIT/ROLLBACK statements.
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys = ON")

        self._create_schema()
        self.generation = self._load_generation()

    # ------------------------------------------------------------------
    # schema and bookkeeping
    # ------------------------------------------------------------------
    def _create_schema(self):
        statements = [
            """
            CREATE TABLE IF NOT EXISTS mdm_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_generation (
                generation INTEGER PRIMARY KEY
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_record_snapshot (
                generation INTEGER NOT NULL,
                source TEXT NOT NULL,
                record_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (generation, source, record_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_link_snapshot (
                generation INTEGER NOT NULL,
                source_a TEXT NOT NULL,
                record_id_a TEXT NOT NULL,
                source_b TEXT NOT NULL,
                record_id_b TEXT NOT NULL,
                probability REAL NOT NULL,
                PRIMARY KEY (
                    generation, source_a, record_id_a, source_b, record_id_b
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_entity_snapshot (
                generation INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (generation, entity_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_member_snapshot (
                generation INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                source TEXT NOT NULL,
                record_id TEXT NOT NULL,
                PRIMARY KEY (generation, source, record_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_fact (
                fact_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                record_id TEXT NOT NULL,
                current_entity_id TEXT,
                payload_json TEXT NOT NULL,
                created_generation INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mdm_fact_history (
                fact_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                source TEXT NOT NULL,
                record_id TEXT NOT NULL,
                entity_id TEXT,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (fact_id, generation)
            )
            """,
        ]
        with self._conn:
            for statement in statements:
                self._conn.execute(statement)
            self._conn.execute(
                "INSERT OR IGNORE INTO mdm_meta(key, value) VALUES (?, '0')",
                (_META_GENERATION,),
            )

    def _load_generation(self):
        row = self._conn.execute(
            "SELECT value FROM mdm_meta WHERE key = ?", (_META_GENERATION,)
        ).fetchone()
        return int(row[0])

    def close(self):
        """Close the underlying SQLite connection."""

        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------------
    # input normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_operations(operations):
        if isinstance(operations, pandas.DataFrame):
            frame = operations
        elif isinstance(operations, dict):
            frame = pandas.DataFrame(operations)
        elif isinstance(operations, (list, tuple)):
            frame = pandas.DataFrame.from_records(operations)
        else:
            raise MDMOperationError(
                "operations must be a pandas DataFrame, a list of dicts "
                "or a dict of columns"
            )

        if frame.empty:
            return []

        for required in ("source", "record_id"):
            if required not in frame.columns:
                raise MDMOperationError(f"operations require a '{required}' column")

        if "_op" in frame.columns and "op" in frame.columns:
            raise MDMOperationError("use either 'op' or '_op', not both")
        op_column = "_op" if "_op" in frame.columns else "op"
        if op_column not in frame.columns:
            raise MDMOperationError("operations require an 'op' column (I/U/D)")

        payload_columns = [
            column
            for column in frame.columns
            if column not in ("source", "record_id", "op", "_op")
        ]

        normalized = []
        seen = set()
        for row in frame.to_dict(orient="records"):
            op = row[op_column]
            if not isinstance(op, str) or op.upper() not in VALID_OPERATIONS:
                raise MDMOperationError(
                    f"operation must be one of {VALID_OPERATIONS}, got {op!r}"
                )
            op = op.upper()
            source = _scalar(row["source"])
            record_id = _scalar(row["record_id"])
            key = (source, record_id)
            if key in seen:
                raise MDMOperationError(
                    f"duplicate operation for record key {key!r} in the same batch"
                )
            seen.add(key)

            if op == "D":
                payload = None
            else:
                payload = _sanitize_mapping(
                    {column: row[column] for column in payload_columns}
                )
            normalized.append((op, source, record_id, payload))

        return normalized

    # ------------------------------------------------------------------
    # current state loaders
    # ------------------------------------------------------------------
    def _load_records(self):
        rows = self._conn.execute(
            """
            SELECT source, record_id, payload_json
            FROM mdm_record_snapshot
            WHERE generation = ?
            """,
            (self.generation,),
        ).fetchall()
        return {
            (source, record_id): _loads(payload_json)
            for source, record_id, payload_json in rows
        }

    def _load_members(self):
        rows = self._conn.execute(
            """
            SELECT entity_id, source, record_id
            FROM mdm_member_snapshot
            WHERE generation = ?
            """,
            (self.generation,),
        ).fetchall()
        return {
            (source, record_id): entity_id for entity_id, source, record_id in rows
        }

    def _load_links(self):
        rows = self._conn.execute(
            """
            SELECT source_a, record_id_a, source_b, record_id_b, probability
            FROM mdm_link_snapshot
            WHERE generation = ?
            """,
            (self.generation,),
        ).fetchall()
        return [
            ((source_a, record_id_a), (source_b, record_id_b), probability)
            for source_a, record_id_a, source_b, record_id_b, probability in rows
        ]

    # ------------------------------------------------------------------
    # linkage pipeline (runs entirely outside the write transaction)
    # ------------------------------------------------------------------
    def _build_dataframe(self, records):
        columns = sorted({key for payload in records.values() for key in payload})
        frame = pandas.DataFrame(
            [
                {column: payload.get(column) for column in columns}
                for payload in records.values()
            ],
            index=pandas.RangeIndex(stop=len(records)),
        )
        tokens = list(records.keys())
        token_to_position = {token: position for position, token in enumerate(tokens)}
        return frame, tokens, token_to_position

    def _score_pairs(self, records, affected_tokens):
        if len(records) < 2 or not affected_tokens:
            return {}

        frame, tokens, token_to_position = self._build_dataframe(records)
        # deleted tokens are no longer part of the frame
        affected_positions = {
            token_to_position[token]
            for token in affected_tokens
            if token in token_to_position
        }
        if not affected_positions:
            return {}

        candidate_pairs = self.indexer.index(frame)
        if candidate_pairs.empty:
            return {}

        # only pairs touching a record changed in this batch need scoring;
        # connections between untouched records are carried over as-is.
        positions_left = candidate_pairs.get_level_values(0)
        positions_right = candidate_pairs.get_level_values(1)
        touched = pandas.Series(
            [
                left in affected_positions or right in affected_positions
                for left, right in zip(positions_left, positions_right)
            ],
            index=candidate_pairs,
        )
        candidate_pairs = candidate_pairs[touched.to_numpy()]
        if candidate_pairs.empty:
            return {}

        features = self.compare.compute(candidate_pairs, frame)

        # the classifier is frozen: inference only, never fit()
        probabilities = self.classifier.prob(features)

        edges = {}
        for (position_a, position_b), probability in probabilities.items():
            if float(probability) >= self.threshold:
                token_a = tokens[int(position_a)]
                token_b = tokens[int(position_b)]
                if _key_to_json(*token_a) <= _key_to_json(*token_b):
                    edge = (token_a, token_b)
                else:
                    edge = (token_b, token_a)
                edges[edge] = float(probability)
        return edges

    # ------------------------------------------------------------------
    # public batch processing
    # ------------------------------------------------------------------
    def process_batch(self, operations):
        """Publish one incremental batch of I/U/D operations.

        Parameters
        ----------
        operations : pandas.DataFrame or list of dict
            Records with ``source``, ``record_id`` and ``op`` columns
            (``op`` is one of ``I``, ``U`` or ``D``; ``_op`` is accepted
            as an alias). Any remaining columns form the raw record
            payload. A list of dictionaries with the same keys is also
            accepted.

        Returns
        -------
        dict
            Summary of the published generation: ``generation``,
            ``inserted``, ``updated``, ``deleted``, ``n_records``,
            ``n_entities``, ``n_links`` and ``n_facts_rebound``.

        Raises
        ------
        MDMOperationError
            If the batch is malformed or inconsistent with the current
            state. No data is modified.
        MDMPublishError
            If a constraint, ``pre_commit`` hook or commit fails. The
            previous generation remains current.
        """

        normalized = self._normalize_operations(operations)
        if not normalized:
            return {
                "generation": self.generation,
                "inserted": 0,
                "updated": 0,
                "deleted": 0,
                "n_records": len(self._load_records()),
                "n_entities": self._count_entities(),
                "n_links": self._count_links(),
                "n_facts_rebound": 0,
            }

        current_records = self._load_records()
        old_members = self._load_members()
        old_links = self._load_links()

        inserted = updated = deleted = 0
        affected_tokens = set()
        for op, source, record_id, payload in normalized:
            key = (source, record_id)
            exists = key in current_records
            if op == "I":
                if exists:
                    raise MDMOperationError(
                        f"cannot insert existing record {key!r}, use 'U'"
                    )
                current_records[key] = payload
                inserted += 1
                affected_tokens.add(key)
            elif op == "U":
                if not exists:
                    raise MDMOperationError(
                        f"cannot update missing record {key!r}, use 'I'"
                    )
                current_records[key] = payload
                updated += 1
                affected_tokens.add(key)
            else:  # D
                if not exists:
                    raise MDMOperationError(
                        f"cannot delete missing record {key!r}"
                    )
                del current_records[key]
                deleted += 1

        # run the real linkage pipeline on the in-memory state; failures
        # here happen before any write, so the old generation is untouched.
        new_edges = self._score_pairs(current_records, affected_tokens)

        current_token_set = set(current_records)
        retained_links = []
        for token_a, token_b, probability in old_links:
            if token_a in current_token_set and token_b in current_token_set:
                if token_a not in affected_tokens and token_b not in affected_tokens:
                    retained_links.append((token_a, token_b, probability))

        all_edges = {
            (token_a, token_b): probability
            for token_a, token_b, probability in retained_links
        }
        all_edges.update(new_edges)

        # connected components; isolated records are singleton components
        union_find = _UnionFind(current_token_set)
        for token_a, token_b in all_edges:
            union_find.union(token_a, token_b)

        new_members = {}
        entities = []
        for component in union_find.components():
            ordered = sorted(component, key=lambda token: _key_to_json(*token))

            # old entity identifiers whose anchor member survives in this
            # component: a merge keeps the smallest old anchor, while a
            # split only inherits into the subgroup holding the anchor.
            inherited = sorted(
                {
                    old_members[token]
                    for token in component
                    if token in old_members
                    and _loads(old_members[token]) == list(token)
                }
            )
            if inherited:
                entity_id = inherited[0]
            else:
                entity_id = _key_to_json(*ordered[0])

            entities.append(entity_id)
            for token in component:
                new_members[token] = entity_id

        new_generation = self.generation + 1
        facts_rebound = self._plan_fact_rebinding(current_token_set, new_members)

        self._publish(
            new_generation,
            current_records,
            all_edges,
            entities,
            new_members,
            facts_rebound,
        )
        self.generation = new_generation

        return {
            "generation": new_generation,
            "inserted": inserted,
            "updated": updated,
            "deleted": deleted,
            "n_records": len(current_records),
            "n_entities": len(entities),
            "n_links": len(all_edges),
            "n_facts_rebound": len(facts_rebound),
        }

    def _plan_fact_rebinding(self, current_token_set, new_members):
        rows = self._conn.execute(
            "SELECT fact_id, source, record_id, current_entity_id FROM mdm_fact"
        ).fetchall()
        planned = []
        for fact_id, source, record_id, old_entity_id in rows:
            key = (source, record_id)
            new_entity_id = (
                new_members[key] if key in current_token_set else None
            )
            if new_entity_id != old_entity_id:
                planned.append((fact_id, source, record_id, new_entity_id))
        return planned

    def _publish(
        self,
        generation,
        records,
        edges,
        entities,
        members,
        facts_rebound,
    ):
        conn = self._conn
        try:
            conn.execute("BEGIN")

            conn.execute(
                "INSERT INTO mdm_generation(generation) VALUES (?)", (generation,)
            )

            conn.executemany(
                """
                INSERT INTO mdm_record_snapshot(
                    generation, source, record_id, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        generation,
                        source,
                        record_id,
                        _dumps(payload),
                    )
                    for (source, record_id), payload in sorted(
                        records.items(), key=lambda item: _key_to_json(*item[0])
                    )
                ],
            )

            conn.executemany(
                """
                INSERT INTO mdm_link_snapshot(
                    generation, source_a, record_id_a, source_b, record_id_b,
                    probability
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        generation,
                        token_a[0],
                        token_a[1],
                        token_b[0],
                        token_b[1],
                        probability,
                    )
                    for (token_a, token_b), probability in sorted(edges.items())
                ],
            )

            conn.executemany(
                "INSERT INTO mdm_entity_snapshot(generation, entity_id) VALUES (?, ?)",
                [(generation, entity_id) for entity_id in sorted(set(entities))],
            )

            conn.executemany(
                """
                INSERT INTO mdm_member_snapshot(
                    generation, entity_id, source, record_id
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (generation, entity_id, source, record_id)
                    for (source, record_id), entity_id in sorted(
                        members.items(), key=lambda item: _key_to_json(*item[0])
                    )
                ],
            )

            for fact_id, source, record_id, new_entity_id in facts_rebound:
                conn.execute(
                    """
                    UPDATE mdm_fact
                    SET current_entity_id = ?
                    WHERE fact_id = ?
                    """,
                    (new_entity_id, fact_id),
                )
                payload_json = conn.execute(
                    "SELECT payload_json FROM mdm_fact WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()[0]
                conn.execute(
                    """
                    INSERT INTO mdm_fact_history(
                        fact_id, generation, source, record_id, entity_id,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact_id,
                        generation,
                        source,
                        record_id,
                        new_entity_id,
                        payload_json,
                    ),
                )

            conn.execute(
                "UPDATE mdm_meta SET value = ? WHERE key = ?",
                (str(generation), _META_GENERATION),
            )

            if self.pre_commit is not None:
                self.pre_commit(conn)

            conn.execute("COMMIT")
        except Exception as err:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(err, MDMError):
                raise
            raise MDMPublishError(
                f"batch publish failed, generation {self.generation} kept"
            ) from err

    # ------------------------------------------------------------------
    # facts
    # ------------------------------------------------------------------
    def add_facts(self, facts):
        """Register or replace facts referencing source records.

        Facts are independent of linkage batches. Their payload never
        changes during rebinding; only ``current_entity_id`` is maintained.
        A fact referencing an unknown or deleted source record gets a
        ``NULL`` entity reference until the record exists.

        Parameters
        ----------
        facts : pandas.DataFrame or list of dict
            Rows with ``fact_id``, ``source``, ``record_id`` and an
            optional ``payload`` mapping.
        """

        if isinstance(facts, pandas.DataFrame):
            rows = facts.to_dict(orient="records")
        elif isinstance(facts, dict):
            rows = pandas.DataFrame(facts).to_dict(orient="records")
        elif isinstance(facts, (list, tuple)):
            rows = list(facts)
        else:
            raise MDMOperationError(
                "facts must be a pandas DataFrame, a list of dicts "
                "or a dict of columns"
            )

        current_members = self._load_members()
        current_token_set = set(current_members)

        conn = self._conn
        try:
            conn.execute("BEGIN")
            for row in rows:
                if "fact_id" not in row or "source" not in row or (
                    "record_id" not in row
                ):
                    raise MDMOperationError(
                        "each fact requires fact_id, source and record_id"
                    )
                fact_id = _scalar(row["fact_id"])
                source = _scalar(row["source"])
                record_id = _scalar(row["record_id"])
                payload = _sanitize_mapping(row.get("payload") or {})
                key = (source, record_id)
                if key in current_token_set:
                    entity_id = current_members[key]
                else:
                    entity_id = None

                conn.execute(
                    """
                    INSERT INTO mdm_fact(
                        fact_id, source, record_id, current_entity_id,
                        payload_json, created_generation
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fact_id) DO UPDATE SET
                        source = excluded.source,
                        record_id = excluded.record_id,
                        current_entity_id = excluded.current_entity_id,
                        payload_json = excluded.payload_json
                    """,
                    (
                        fact_id,
                        source,
                        record_id,
                        entity_id,
                        _dumps(payload),
                        self.generation,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    # ------------------------------------------------------------------
    # read-only views of the current generation
    # ------------------------------------------------------------------
    def _count_entities(self):
        return self._conn.execute(
            "SELECT COUNT(*) FROM mdm_entity_snapshot WHERE generation = ?",
            (self.generation,),
        ).fetchone()[0]

    def _count_links(self):
        return self._conn.execute(
            "SELECT COUNT(*) FROM mdm_link_snapshot WHERE generation = ?",
            (self.generation,),
        ).fetchone()[0]

    def get_records(self):
        """Return the raw records of the current generation."""

        rows = self._conn.execute(
            """
            SELECT source, record_id, payload_json
            FROM mdm_record_snapshot
            WHERE generation = ?
            ORDER BY source, record_id
            """,
            (self.generation,),
        ).fetchall()
        return pandas.DataFrame(
            [
                {
                    "source": source,
                    "record_id": record_id,
                    "payload": _loads(payload_json),
                }
                for source, record_id, payload_json in rows
            ],
            columns=["source", "record_id", "payload"],
        )

    def get_links(self):
        """Return the accepted connections of the current generation."""

        rows = self._conn.execute(
            """
            SELECT source_a, record_id_a, source_b, record_id_b, probability
            FROM mdm_link_snapshot
            WHERE generation = ?
            ORDER BY source_a, record_id_a, source_b, record_id_b
            """,
            (self.generation,),
        ).fetchall()
        return pandas.DataFrame(
            rows,
            columns=[
                "source_a",
                "record_id_a",
                "source_b",
                "record_id_b",
                "probability",
            ],
        )

    def get_entities(self):
        """Return the entities of the current generation."""

        rows = self._conn.execute(
            """
            SELECT e.entity_id, COUNT(m.source) AS n_members
            FROM mdm_entity_snapshot e
            LEFT JOIN mdm_member_snapshot m
                ON m.generation = e.generation
                AND m.entity_id = e.entity_id
            WHERE e.generation = ?
            GROUP BY e.entity_id
            ORDER BY e.entity_id
            """,
            (self.generation,),
        ).fetchall()
        frame = pandas.DataFrame(rows, columns=["entity_id", "n_members"])
        if not frame.empty:
            anchors = frame["entity_id"].map(_loads)
            frame["anchor_source"] = [anchor[0] for anchor in anchors]
            frame["anchor_record_id"] = [anchor[1] for anchor in anchors]
        else:
            frame["anchor_source"] = []
            frame["anchor_record_id"] = []
        return frame[
            ["entity_id", "anchor_source", "anchor_record_id", "n_members"]
        ]

    def get_members(self):
        """Return the entity membership of every record."""

        rows = self._conn.execute(
            """
            SELECT entity_id, source, record_id
            FROM mdm_member_snapshot
            WHERE generation = ?
            ORDER BY entity_id, source, record_id
            """,
            (self.generation,),
        ).fetchall()
        return pandas.DataFrame(rows, columns=["entity_id", "source", "record_id"])

    def get_facts(self):
        """Return the facts and their current entity reference."""

        rows = self._conn.execute(
            """
            SELECT fact_id, source, record_id, current_entity_id, payload_json
            FROM mdm_fact
            ORDER BY fact_id
            """
        ).fetchall()
        return pandas.DataFrame(
            [
                {
                    "fact_id": fact_id,
                    "source": source,
                    "record_id": record_id,
                    "current_entity_id": current_entity_id,
                    "payload": _loads(payload_json),
                }
                for fact_id, source, record_id, current_entity_id, payload_json in rows
            ],
            columns=[
                "fact_id",
                "source",
                "record_id",
                "current_entity_id",
                "payload",
            ],
        )

    def get_fact_history(self, fact_id=None):
        """Return the per-generation rebinding history of facts."""

        if fact_id is None:
            rows = self._conn.execute(
                """
                SELECT fact_id, generation, source, record_id, entity_id,
                       payload_json
                FROM mdm_fact_history
                ORDER BY fact_id, generation
                """
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT fact_id, generation, source, record_id, entity_id,
                       payload_json
                FROM mdm_fact_history
                WHERE fact_id = ?
                ORDER BY generation
                """,
                (fact_id,),
            ).fetchall()
        return pandas.DataFrame(
            [
                {
                    "fact_id": fid,
                    "generation": generation,
                    "source": source,
                    "record_id": record_id,
                    "entity_id": entity_id,
                    "payload": _loads(payload_json),
                }
                for fid, generation, source, record_id, entity_id, payload_json in rows
            ],
            columns=[
                "fact_id",
                "generation",
                "source",
                "record_id",
                "entity_id",
                "payload",
            ],
        )
