"""vmstat delta semantics: traffic/reclaim counters, None-preservation."""

from osrl.vmstat import metric_deltas, metric_values, read_vmstat

BASE = {
    "pswpin": 157141,
    "pswpout": 1026156,
    "pgmajfault": 410768,
    "pgscan_kswapd": 6242614,
    "pgscan_direct": 228119,
    "pgsteal_kswapd": 4970274,
    "allocstall_normal": 1070,
    "allocstall_movable": 763,
}


def write(tmp_path, counters):
    path = tmp_path / "vmstat"
    path.write_text("\n".join(f"{k} {v}" for k, v in counters.items()) + "\n")
    return read_vmstat(path)


def test_metric_values_sum_prefixes(tmp_path):
    snap = write(tmp_path, BASE)
    vals = metric_values(snap)
    assert vals["swp_in"] == 157141
    assert vals["swp_out"] == 1_026_156
    assert vals["majflt"] == 410_768
    assert vals["reclaim_scan"] == 6_242_614 + 228_119
    assert vals["reclaim_steal"] == 4_970_274
    assert vals["allocstall"] == 1070 + 763


def test_missing_counter_is_none_not_zero(tmp_path):
    snap = write(tmp_path, {"pswpin": 10})  # pswpout etc absent
    vals = metric_values(snap)
    assert vals["swp_in"] == 10
    assert vals["swp_out"] is None
    assert vals["allocstall"] is None


def test_first_tick_deltas_are_none(tmp_path):
    snap = write(tmp_path, BASE)
    deltas = metric_deltas(None, snap)
    assert all(v is None for v in deltas.values())


def test_deltas_and_reset(tmp_path):
    prev = write(tmp_path, BASE)
    cur = write(tmp_path, {k: v + 120 for k, v in BASE.items()})
    d = metric_deltas(prev, cur)
    assert d["swp_out"] == 120
    # prefix metric sums every allocstall_* zone counter: 2 counters x 120
    assert d["allocstall"] == 240
    reset = write(tmp_path, {k: v - 5000 for k, v in BASE.items()})
    d = metric_deltas(prev, reset)
    assert all(v is None for v in d.values())  # reboot is never a negative rate


def test_unreadable_file_is_none(tmp_path):
    assert read_vmstat(tmp_path / "nope") is None
