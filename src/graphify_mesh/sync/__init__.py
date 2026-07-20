"""graphify-mesh sync engine: inventory + sync pipeline.

Merge-semantics decision (the single most load-bearing choice in this engine):

    `graphify merge-graphs <g1> <g2> ... --out <path>` is STATELESS — it loads
    each graph.json fresh, normalizes Graph/DiGraph/MultiGraph to a plain
    nx.Graph, prefixes node ids with a distinct per-repo tag
    (graphify's cli.py / build.py:distinct_repo_tags), and composes via
    nx.compose into a brand-new graph. No manifest, no incremental state, no
    external-node dedup-by-label.

    `graphify global add <graph> --as <tag>` is STATEFUL — it keeps
    ~/.graphify/global-manifest.json and, on each add, deduplicates "external"
    nodes (nodes with no source_file) by label against whatever is already in
    the accumulated global graph, remapping edges from the newly-added repo
    onto the *first-added* repo's copy of that node. Re-adding a repo out of
    order, or pruning+re-adding one repo, silently rewires or drops edges that
    other repos depend on for shared external nodes — this is the identity
    corruption this engine exists to avoid.

    DECISION: this sync engine (graphify_mesh.sync.publish /
    graphify_mesh.sync.pipeline) always calls `graphify merge-graphs` with a
    deterministic, sorted-by-repo_id list of per-repo graph.json paths,
    rebuilding the global graph from empty on every run. `graphify global add`
    is never invoked anywhere in this package.
"""
