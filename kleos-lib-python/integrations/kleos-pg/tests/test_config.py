from kleos_pg import config_yaml


def test_config_yaml():
    y = config_yaml("crm", "postgres://u:${PW}@h/db", writeback_allow=["public.tickets"])
    assert "source: crm" in y
    assert 'connection: "postgres://u:${PW}@h/db"' in y
    assert 'writeback: {allow: ["public.tickets"]}' in y
