"""PSI parsing + interval arithmetic, checked against the kernel contract."""

import pytest

from osrl.psi import interval_stall_us, parse_psi_text, read_psi

REALISTIC = (
    "some avg10=0.03 avg60=0.07 avg300=0.05 total=48508075\n"
    "full avg10=0.00 avg60=0.00 avg300=0.00 total=1127882\n"
)


def test_parse_real_format():
    stats = parse_psi_text(REALISTIC)
    assert set(stats) == {"some", "full"}
    some = stats["some"]
    assert some.avg10 == pytest.approx(0.03)
    assert some.avg60 == pytest.approx(0.07)
    assert some.avg300 == pytest.approx(0.05)
    assert some.total_us == 48_508_075  # cumulative microseconds, kept raw
    assert stats["full"].total_us == 1_127_882


def test_parse_cpu_full_documented_zero():
    # kernel docs: system-level CPU 'full' is undefined and reported as zero
    text = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n" \
           "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    stats = parse_psi_text(text)
    assert stats["full"].total_us == 0  # parseable; consumers must not use it


def test_parse_rejects_malformed_line():
    with pytest.raises(ValueError):
        parse_psi_text("some avg10=broken total=5\n")


def test_read_psi_missing_file_returns_none(tmp_path):
    assert read_psi(tmp_path / "cpu") is None


def test_read_psi_malformed_file_returns_none(tmp_path):
    p = tmp_path / "memory"
    p.write_text("garbage line without fields\n")
    assert read_psi(p) is None


def test_interval_stall_is_a_diff_not_a_raw_total():
    assert interval_stall_us(1_000_000, 1_250_000) == 250_000


def test_interval_stall_reset_is_none_not_zero():
    # counters restarted (reboot); fabricating a stall here would be a lie
    assert interval_stall_us(1_500_000, 200) is None


def test_interval_stall_unknown_endpoint_is_none():
    assert interval_stall_us(None, 1_000) is None
    assert interval_stall_us(1_000, None) is None


def test_interval_stall_zero_is_valid_when_no_new_stall():
    assert interval_stall_us(2000, 2000) == 0
