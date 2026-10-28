import logging

from src.main import configure_logging


def test_unknown_log_level_falls_back_to_info_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        configure_logging("VERBOSE")

    assert "Unknown LOG_LEVEL 'VERBOSE', logging at INFO; use DEBUG, INFO, WARNING, ERROR or CRITICAL" in caplog.text


def test_known_log_level_is_set_without_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        configure_logging("DEBUG")

    assert caplog.text == ""
