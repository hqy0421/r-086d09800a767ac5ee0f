#!/usr/bin/env python

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import recordlinkage as rl
from recordlinkage.mdm import MDMManager
from recordlinkage.mdm import MDMOperationError
from recordlinkage.mdm import MDMPublishError
from recordlinkage.mdm import _key_to_json


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _build_pipeline(threshold_link=0.85):
    indexer = rl.Index()
    indexer.block(left_on="city")

    compare = rl.Compare()
    compare.string(
        "name", "name", method="jarowinkler", label="name_sim"
    )
    compare.exact("city", "city", label="city_exact")

    # fitted once, then frozen. Fit the sklearn kernel directly with a
    # labelled feature matrix (the high-level fit() expects pair-indexed
    # comparison vectors).
    training = pd.DataFrame(
        {
            "name_sim": [0.0, 0.25, 0.4, 0.9, 0.95, 1.0],
            "city_exact": [0, 0, 1, 1, 1, 1],
        }
    )
    labels = [0, 0, 0, 1, 1, 1]
    classifier = rl.LogisticRegressionClassifier()
    classifier.kernel.fit(training.values, labels)
    classifier.kernel.classes_ = np.array([0, 1])
    return indexer, compare, classifier


@pytest.fixture()
def pipeline():
    return _build_pipeline()


@pytest.fixture()
def manager(tmp_path, pipeline):
    indexer, compare, classifier = pipeline
    mdm = MDMManager(
        tmp_path / "mdm.sqlite",
        indexer=indexer,
        compare=compare,
        classifier=classifier,
        threshold=0.5,
    )
    yield mdm
    mdm.close()


def _batch(*rows):
    return pd.DataFrame(rows)


def _entity_of(members_frame, source, record_id):
    mask = (members_frame["source"] == source) & (
        members_frame["record_id"] == record_id
    )
    values = members_frame.loc[mask, "entity_id"]
    assert len(values) == 1
    return values.iloc[0]


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------
def test_construction_validates_arguments(tmp_path, pipeline):
    indexer, compare, classifier = pipeline

    with pytest.raises(TypeError):
        MDMManager(tmp_path / "x.db", indexer="no", compare=compare,
                   classifier=classifier)
    with pytest.raises(TypeError):
        MDMManager(tmp_path / "x.db", indexer=indexer, compare="no",
                   classifier=classifier)
    with pytest.raises(TypeError):
        MDMManager(tmp_path / "x.db", indexer=indexer, compare=compare,
                   classifier=object())
    with pytest.raises(ValueError):
        MDMManager(
            tmp_path / "x.db", indexer, compare, classifier, threshold=1.5
        )
    with pytest.raises(TypeError):
        MDMManager(
            tmp_path / "x.db", indexer, compare, classifier, pre_commit=42
        )


def test_malformed_operations(manager):
    with pytest.raises(MDMOperationError):
        manager.process_batch(
            pd.DataFrame({"source": ["s1"], "record_id": ["r1"]})
        )  # no op column

    with pytest.raises(MDMOperationError):
        manager.process_batch(
            _batch(
                dict(source="s1", record_id="r1", op="X"),
            )
        )

    with pytest.raises(MDMOperationError):
        manager.process_batch(
            [
                dict(source="s1", record_id="r1", op="I"),
                dict(source="s1", record_id="r1", op="I"),
            ]
        )

    with pytest.raises(MDMOperationError):
        manager.process_batch(
            _batch(dict(source="s1", record_id="r1", op="U"))
        )  # update missing

    with pytest.raises(MDMOperationError):
        manager.process_batch(
            _batch(dict(source="s1", record_id="r1", op="D"))
        )  # delete missing

    manager.process_batch(
        _batch(dict(source="s1", record_id="r1", op="I", name="a", city="c"))
    )
    with pytest.raises(MDMOperationError):
        manager.process_batch(
            _batch(dict(source="s1", record_id="r1", op="I", name="a", city="c"))
        )


def test_operations_accept_list_and_dict(manager):
    manager.process_batch(
        [
            {"source": "s1", "record_id": 1, "op": "I", "name": "John", "city": "NY"},
            {"source": "s2", "record_id": "a", "op": "i", "name": "Jane", "city": "LA"},
        ]
    )
    assert manager.generation == 1
    assert manager.get_records().shape[0] == 2


# ---------------------------------------------------------------------------
# isolated records become singleton entities
# ---------------------------------------------------------------------------
def test_isolated_records_become_entities(manager):
    result = manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Jane", city="LA"),
        )
    )

    assert result == {
        "generation": 1,
        "inserted": 2,
        "updated": 0,
        "deleted": 0,
        "n_records": 2,
        "n_entities": 2,
        "n_links": 0,
        "n_facts_rebound": 0,
    }

    entities = manager.get_entities()
    assert sorted(entities["n_members"]) == [1, 1]
    expected_ids = sorted(
        [_key_to_json("s1", "r1"), _key_to_json("s2", "r2")]
    )
    assert list(entities["entity_id"]) == expected_ids

    members = manager.get_members()
    assert _entity_of(members, "s1", "r1") == _key_to_json("s1", "r1")
    assert manager.get_links().empty


# ---------------------------------------------------------------------------
# merge keeps the smallest old anchor entity id
# ---------------------------------------------------------------------------
def test_merge_keeps_smallest_old_anchor(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John Smith", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Jane Doe", city="LA"),
        )
    )
    id_r1 = _key_to_json("s1", "r1")
    id_r2 = _key_to_json("s2", "r2")

    # a near-duplicate of r1 from a third source merges into r1's entity
    manager.process_batch(
        _batch(
            dict(
                source="s3",
                record_id="r3",
                op="I",
                name="John Smithh",
                city="NY",
            )
        )
    )
    members = manager.get_members()
    merged_id = _entity_of(members, "s1", "r1")
    assert merged_id == id_r1
    assert _entity_of(members, "s3", "r3") == merged_id
    assert _entity_of(members, "s2", "r2") == id_r2
    assert manager.get_entities().shape[0] == 2
    assert manager.get_links().shape[0] == 1

    # now connect both old entities through a bridging record -> merge keeps
    # the smallest old anchor identifier
    manager.process_batch(
        _batch(
            dict(
                source="s4",
                record_id="r4",
                op="I",
                name="Jane Doee",
                city="LA",
            )
        )
    )
    # r4 merges with r2 first; force the two entities together by updating
    # r4 so it also matches the r1 entity (same city, close name)
    manager.process_batch(
        _batch(
            dict(
                source="s4",
                record_id="r4",
                op="U",
                name="Johnn Smith",
                city="NY",
            )
        )
    )

    members = manager.get_members()
    final_ids = set(members["entity_id"])
    # one merged entity (id_r1 is the smallest anchor) plus no orphans
    assert id_r1 in final_ids
    merged_members = members[members["entity_id"] == id_r1]
    assert set(
        zip(merged_members["source"], merged_members["record_id"])
    ) == {("s1", "r1"), ("s3", "r3"), ("s4", "r4")}
    # the r2 entity stays separate because r4 moved away
    assert _entity_of(members, "s2", "r2") == id_r2


def test_merge_two_entities_smallest_anchor(manager):
    # directly engineer a two-entity merge with a chain in one batch
    manager.process_batch(
        _batch(
            dict(source="a", record_id="1", op="I", name="John", city="NY"),
            dict(source="b", record_id="2", op="I", name="Mary", city="LA"),
        )
    )
    id_a = _key_to_json("a", "1")
    id_b = _key_to_json("b", "2")
    assert id_a < id_b

    # update b's record so it links to a (same city, close name)
    manager.process_batch(
        _batch(dict(source="b", record_id="2", op="U", name="Johnn", city="NY"))
    )

    members = manager.get_members()
    assert set(members["entity_id"]) == {id_a}
    assert _entity_of(members, "b", "2") == id_a


# ---------------------------------------------------------------------------
# split: anchor subgroup inherits, others take smallest member
# ---------------------------------------------------------------------------
def test_split_anchor_inheritance(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
        )
    )
    manager.process_batch(
        _batch(
            dict(source="s2", record_id="r2", op="I", name="Johnn", city="NY"),
            dict(source="s3", record_id="r3", op="I", name="Johnh", city="NY"),
        )
    )
    members = manager.get_members()
    anchor_id = _key_to_json("s1", "r1")
    assert len(set(members["entity_id"])) == 1
    assert members["entity_id"].iloc[0] == anchor_id

    # break the connections by moving r2/r3 to a distant city (blocking key
    # changes, so no edges between them and r1 survive)
    manager.process_batch(
        _batch(
            dict(source="s2", record_id="r2", op="U", name="Mary", city="LA"),
            dict(source="s3", record_id="r3", op="U", name="Maria", city="LA"),
        )
    )

    members = manager.get_members()
    # r1 keeps the old anchor id; r2 and r3 may stay linked to each other or
    # split, but never inherit r1's id
    assert _entity_of(members, "s1", "r1") == anchor_id
    remaining_ids = set(members["entity_id"]) - {anchor_id}
    for entity_id in remaining_ids:
        subset = members[members["entity_id"] == entity_id]
        assert not (
            (subset["source"] == "s1") & (subset["record_id"] == "r1")
        ).any()
        # a non-anchor id equals the smallest member of its subgroup
        smallest = min(
            _key_to_json(s, r)
            for s, r in zip(subset["source"], subset["record_id"])
        )
        assert entity_id == smallest


def test_split_after_anchor_deleted_uses_smallest_member(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Johnn", city="NY"),
        )
    )
    anchor_id = _key_to_json("s1", "r1")
    members = manager.get_members()
    assert set(members["entity_id"]) == {anchor_id}

    # delete the anchor; the surviving record takes its own (smallest) key
    manager.process_batch(_batch(dict(source="s1", record_id="r1", op="D")))

    members = manager.get_members()
    assert _entity_of(members, "s2", "r2") == _key_to_json("s2", "r2")
    entities = manager.get_entities()
    assert anchor_id not in set(entities["entity_id"])


# ---------------------------------------------------------------------------
# deletes
# ---------------------------------------------------------------------------
def test_delete_removes_record_and_singleton_entity(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Jane", city="LA"),
        )
    )
    manager.process_batch(_batch(dict(source="s1", record_id="r1", op="D")))

    records = manager.get_records()
    assert list(zip(records["source"], records["record_id"])) == [("s2", "r2")]
    assert manager.get_members().shape[0] == 1
    assert manager.get_entities().shape[0] == 1
    assert manager.get_links().empty


def test_delete_only_batch_with_survivors(manager):
    # every affected token is a deletion; the surviving records must still
    # be rebuilt from retained connections without a KeyError
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Johnn", city="NY"),
            dict(source="s3", record_id="r3", op="I", name="Jane", city="LA"),
        )
    )
    assert manager.get_links().shape[0] == 1

    result = manager.process_batch(
        _batch(dict(source="s3", record_id="r3", op="D"))
    )
    assert result["deleted"] == 1
    assert result["n_records"] == 2
    assert result["n_entities"] == 1
    assert manager.get_links().shape[0] == 1


def test_delete_all_records(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Jane", city="LA"),
        )
    )
    result = manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="D"),
            dict(source="s2", record_id="r2", op="D"),
        )
    )
    assert result["n_records"] == 0
    assert result["n_entities"] == 0
    assert result["n_links"] == 0
    assert manager.get_records().empty
    assert manager.get_members().empty
    assert manager.get_entities().empty
    assert manager.get_links().empty


# ---------------------------------------------------------------------------
# facts: rebind on merge, NULL on delete, payload untouched
# ---------------------------------------------------------------------------
def test_facts_rebind_and_null(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Jane", city="LA"),
        )
    )
    id_r1 = _key_to_json("s1", "r1")
    id_r2 = _key_to_json("s2", "r2")

    manager.add_facts(
        [
            {
                "fact_id": "f1",
                "source": "s1",
                "record_id": "r1",
                "payload": {"amount": 100},
            },
            {
                "fact_id": "f2",
                "source": "s2",
                "record_id": "r2",
                "payload": {"amount": 200},
            },
        ]
    )

    facts = manager.get_facts().set_index("fact_id")
    assert facts.loc["f1", "current_entity_id"] == id_r1
    assert facts.loc["f2", "current_entity_id"] == id_r2

    # merge r2 into r1: f2 must be rebound to the merged entity, payload
    # unchanged
    manager.process_batch(
        _batch(dict(source="s2", record_id="r2", op="U", name="Johnn", city="NY"))
    )
    facts = manager.get_facts().set_index("fact_id")
    assert facts.loc["f1", "current_entity_id"] == id_r1
    assert facts.loc["f2", "current_entity_id"] == id_r1
    assert facts.loc["f2", "payload"] == {"amount": 200}

    history = manager.get_fact_history("f2")
    assert list(history["entity_id"]) == [id_r1]
    assert history["payload"].iloc[0] == {"amount": 200}

    # delete the source record referenced by f2 -> NULL, fact row survives
    manager.process_batch(_batch(dict(source="s2", record_id="r2", op="D")))
    facts = manager.get_facts().set_index("fact_id")
    assert facts.loc["f2", "current_entity_id"] is None
    assert facts.loc["f2", "payload"] == {"amount": 200}
    assert pd.isna(facts.loc["f2", "current_entity_id"])

    history = manager.get_fact_history("f2")
    assert history["entity_id"].tolist() == [id_r1, None]
    # payload never changes
    assert all(payload == {"amount": 200} for payload in history["payload"])


def test_fact_before_record_exists_is_null_then_rebound(manager):
    manager.add_facts(
        [{"fact_id": "f1", "source": "s1", "record_id": "r1", "payload": {}}]
    )
    assert manager.get_facts().iloc[0]["current_entity_id"] is None

    manager.process_batch(
        _batch(dict(source="s1", record_id="r1", op="I", name="John", city="NY"))
    )
    facts = manager.get_facts()
    assert facts.iloc[0]["current_entity_id"] == _key_to_json("s1", "r1")


def test_add_facts_upsert(manager):
    manager.process_batch(
        _batch(dict(source="s1", record_id="r1", op="I", name="John", city="NY"))
    )
    manager.add_facts(
        [{"fact_id": "f1", "source": "s1", "record_id": "r1", "payload": {"v": 1}}]
    )
    manager.add_facts(
        [{"fact_id": "f1", "source": "s1", "record_id": "r1", "payload": {"v": 2}}]
    )
    facts = manager.get_facts()
    assert len(facts) == 1
    assert facts.iloc[0]["payload"] == {"v": 2}


# ---------------------------------------------------------------------------
# atomicity: failure keeps the previous generation
# ---------------------------------------------------------------------------
def test_pre_commit_failure_rolls_back(manager):
    manager.process_batch(
        _batch(dict(source="s1", record_id="r1", op="I", name="John", city="NY"))
    )
    manager.pre_commit = lambda conn: (_ for _ in ()).throw(
        RuntimeError("constraint violation")
    )

    with pytest.raises(MDMPublishError):
        manager.process_batch(
            _batch(
                dict(source="s2", record_id="r2", op="I", name="Jane", city="LA")
            )
        )

    assert manager.generation == 1
    assert manager.get_records().shape[0] == 1
    assert manager.get_entities().shape[0] == 1
    # a failed generation must leave no snapshot rows behind
    conn = sqlite3.connect(manager.path)
    assert conn.execute(
        "SELECT DISTINCT generation FROM mdm_record_snapshot"
    ).fetchall() == [(1,)]
    conn.close()


def test_scoring_failure_leaves_state_untouched(manager):
    manager.process_batch(
        _batch(dict(source="s1", record_id="r1", op="I", name="John", city="NY"))
    )

    def broken_prob(features):
        raise RuntimeError("model blew up")

    manager.classifier.prob = broken_prob

    with pytest.raises(RuntimeError):
        manager.process_batch(
            _batch(
                dict(source="s2", record_id="r2", op="I", name="Johnn", city="NY")
            )
        )
    assert manager.generation == 1
    assert manager.get_records().shape[0] == 1


def test_reopen_continues_from_latest_generation(tmp_path, pipeline):
    indexer, compare, classifier = pipeline
    path = tmp_path / "mdm.sqlite"

    mdm = MDMManager(path, indexer, compare, classifier, threshold=0.5)
    mdm.process_batch(
        _batch(dict(source="s1", record_id="r1", op="I", name="John", city="NY"))
    )
    mdm.close()

    reopened = MDMManager(path, indexer, compare, classifier, threshold=0.5)
    try:
        assert reopened.generation == 1
        assert reopened.get_records().shape[0] == 1
        assert reopened.get_entities().shape[0] == 1

        # processing continues with the next generation number
        reopened.process_batch(
            _batch(
                dict(source="s2", record_id="r2", op="I", name="Jane", city="LA")
            )
        )
        assert reopened.generation == 2
        assert reopened.get_records().shape[0] == 2
    finally:
        reopened.close()


def test_generations_are_kept_as_history(tmp_path, pipeline):
    indexer, compare, classifier = pipeline
    mdm = MDMManager(tmp_path / "m.db", indexer, compare, classifier)
    mdm.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Jane", city="LA"),
        )
    )
    mdm.process_batch(_batch(dict(source="s2", record_id="r2", op="D")))

    conn = sqlite3.connect(mdm.path)
    gen_counts = dict(
        conn.execute(
            "SELECT generation, COUNT(*) FROM mdm_record_snapshot GROUP BY generation"
        ).fetchall()
    )
    assert gen_counts == {1: 2, 2: 1}
    entity_counts = dict(
        conn.execute(
            "SELECT generation, COUNT(*) FROM mdm_entity_snapshot GROUP BY generation"
        ).fetchall()
    )
    assert entity_counts == {1: 2, 2: 1}
    conn.close()
    mdm.close()


# ---------------------------------------------------------------------------
# the model is frozen: only prob() is invoked
# ---------------------------------------------------------------------------
def test_model_is_not_refit(manager, pipeline):
    _, _, classifier = pipeline
    before = classifier.kernel.coef_.copy()
    intercept_before = classifier.kernel.intercept_.copy()

    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Johnn", city="NY"),
        )
    )
    manager.process_batch(
        _batch(
            dict(source="s3", record_id="r3", op="I", name="Johnh", city="NY"),
            dict(source="s4", record_id="r4", op="I", name="Mary", city="LA"),
        )
    )

    np.testing.assert_array_equal(classifier.kernel.coef_, before)
    np.testing.assert_array_equal(classifier.kernel.intercept_, intercept_before)


def test_real_index_compare_and_prob_are_used(manager):
    # records sharing a blocking key with very different names must not link;
    # the threshold and the fitted model drive the decision
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Xavier", city="NY"),
        )
    )
    assert manager.get_entities().shape[0] == 2
    assert manager.get_links().empty

    manager.process_batch(
        _batch(
            dict(source="s3", record_id="r3", op="I", name="Johnn", city="NY"),
        )
    )
    members = manager.get_members()
    assert _entity_of(members, "s3", "r3") == _entity_of(members, "s1", "r1")
    links = manager.get_links()
    assert links.shape[0] == 1
    assert 0.0 <= float(links["probability"].iloc[0]) <= 1.0


def test_untouched_connections_survive(manager):
    manager.process_batch(
        _batch(
            dict(source="s1", record_id="r1", op="I", name="John", city="NY"),
            dict(source="s2", record_id="r2", op="I", name="Johnn", city="NY"),
            dict(source="s3", record_id="r3", op="I", name="Jane", city="LA"),
        )
    )
    assert manager.get_links().shape[0] == 1

    # an unrelated update must not drop the existing r1-r2 connection
    manager.process_batch(
        _batch(dict(source="s3", record_id="r3", op="U", name="Janet", city="LA"))
    )
    links = manager.get_links()
    assert links.shape[0] == 1
    assert (links["source_a"].iloc[0], links["record_id_a"].iloc[0]) == ("s1", "r1")
    assert (links["source_b"].iloc[0], links["record_id_b"].iloc[0]) == ("s2", "r2")


def test_empty_batch_is_noop(manager):
    result = manager.process_batch(pd.DataFrame(columns=["source", "record_id", "op"]))
    assert result["generation"] == 0
    assert manager.generation == 0


def test_payload_persisted_as_raw_json(manager):
    manager.process_batch(
        _batch(
            dict(
                source="s1",
                record_id="r1",
                op="I",
                name="John",
                city="NY",
                score=3.5,
            )
        )
    )
    record = manager.get_records().iloc[0]
    assert record["payload"] == {"name": "John", "city": "NY", "score": 3.5}

    conn = sqlite3.connect(manager.path)
    raw = conn.execute(
        "SELECT payload_json FROM mdm_record_snapshot WHERE generation = 1"
    ).fetchone()[0]
    assert json.loads(raw)["score"] == 3.5
    conn.close()
