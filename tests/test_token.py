"""Where the Earthdata token is read from."""
import pytest

from asf_cleaner import config


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.setattr(config, "ENV_FILES", (tmp_path / "here.env", tmp_path / "project.env"))
    return tmp_path


def test_missing_token_says_where_to_put_it():
    with pytest.raises(RuntimeError, match=r"\.env"):
        config.get_token()


def test_variable_wins(monkeypatch, clean):
    (clean / "here.env").write_text("EARTHDATA_TOKEN=from-file\n")
    monkeypatch.setenv("EARTHDATA_TOKEN", "from-env")
    assert config.get_token() == "from-env"


def test_env_file_forms_and_empty_value(clean):
    (clean / "here.env").write_text("# comment\nEARTHDATA_TOKEN=\n")
    (clean / "project.env").write_text("export EARTHDATA_TOKEN='abc.def'\n")
    assert config.get_token() == "abc.def"  # an empty line is not a token; the next file is tried


def test_token_added_while_running_is_picked_up(clean):
    with pytest.raises(RuntimeError):
        config.get_token()
    (clean / "project.env").write_text("EARTHDATA_TOKEN=late\n")
    assert config.get_token() == "late"
