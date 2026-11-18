**************************
Master data management
**************************

The :class:`recordlinkage.MDMManager` builds a multi-source customer
master data layer on top of the standard Index, Compare and
classification pipeline. It accepts incremental insert/update/delete
operations identified by ``(source, record_id)``, runs the *frozen*
linkage model, and persists the resulting master entities and their
members in SQLite.

.. autoclass:: recordlinkage.MDMManager
   :members:

.. autoexception:: recordlinkage.MDMError

.. autoexception:: recordlinkage.MDMOperationError

.. autoexception:: recordlinkage.MDMPublishError

Overview
========

Records arrive as batches of ``I`` (insert), ``U`` (update) and ``D``
(delete) operations. Every batch is processed as follows:

1. The raw records are updated in memory and candidate pairs are produced
   with the configured :class:`recordlinkage.Index` algorithms.
2. The candidate pairs are compared with the configured
   :class:`recordlinkage.Compare` features and scored with the fitted
   probability model. The model is never retrained; only its ``prob``
   method is called.
3. Pairs whose match probability reaches the configured threshold become
   connections. The connected components of the connection graph are the
   current entities; isolated records become singleton entities.
4. Raw records, connections, entities, members and fact references are
   published together in a single SQLite transaction as a new
   generation. If scoring, a constraint or the commit fails, the previous
   generation stays current and reopening the database continues from it.

Entity identifiers are stable:

* a merge keeps the smallest identifier of the merged old anchor
  entities;
* on a split, the subgroup containing the old anchor inherits the
  identifier, every other subgroup takes its smallest member key;
* a new component without an old anchor takes its smallest member key.

Facts reference source records. After every batch their reference to the
current entity is rebound (including across merges and splits), deleting
the referenced source record sets the reference to ``NULL`` and the fact
payload itself is never modified.
