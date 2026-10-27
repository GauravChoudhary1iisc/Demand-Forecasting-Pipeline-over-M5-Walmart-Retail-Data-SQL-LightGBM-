import duckdb
import pytest


@pytest.fixture(scope="session")
def con():
    c = duckdb.connect("m5.duckdb", read_only=True)
    yield c
    c.close()
