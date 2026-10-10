import pytest


@pytest.fixture(autouse=True)
def disable_live_diagnostic_logs(monkeypatch):
    """Tests use explicit temporary files when they intend to inspect logs."""
    monkeypatch.setenv('TRASSIST_CC2_TRACE', 'off')
    monkeypatch.setenv('TRASSIST_CC2_METRICS', 'off')
