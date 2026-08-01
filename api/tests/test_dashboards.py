"""The provisioned Grafana dashboards (spec §9.7).

Every failure mode here is *silent*: a dashboard with bad JSON, a duplicate uid
or a panel that names no datasource renders as an empty board rather than as an
error, which reads like "no data" instead of "broken config". §9.7 already
records two provisioning traps that each cost an evening; these are the cheap
checks that stop a third.

Deliberately NOT asserted: that a query returns rows. That needs a live
Prometheus with traffic in it, which is not something a unit test should own.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
import yaml

OPS = pathlib.Path(__file__).resolve().parents[2] / "ops" / "grafana"
DASHBOARDS = sorted((OPS / "dashboards").glob("*.json"))
KNOWN_DATASOURCES = {"prometheus", "loki", "tempo", "coffee-shop-db"}


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def test_there_is_a_dashboard_for_each_agent_and_one_for_the_shop():
    """One board per role, because "how expensive is Mo" is not a question a
    single mixed board can answer (§13.11)."""
    uids = {load(p)["uid"] for p in DASHBOARDS}

    assert uids == {"sam-waiter", "mo-barista", "val-cashier", "coffee-shop"}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_dashboard_is_valid_and_uniquely_identified(path):
    dashboard = load(path)

    assert dashboard["uid"], "no uid means Grafana invents one on every restart"
    assert dashboard["title"]
    assert dashboard["panels"], "an empty board looks like missing data"
    # The JSON is the source of truth, not the UI — an edit saved in Grafana is
    # reverted on the next scan, which is the behaviour we want.
    assert dashboard["editable"] is False


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_every_panel_and_target_names_a_known_datasource(path):
    """`grafana/otel-lgtm` marks no datasource as default, so a panel that omits
    one renders empty and looks like missing data (§9.7)."""
    for panel in load(path)["panels"]:
        where = f"{path.stem}: {panel['title']}"
        uid = (panel.get("datasource") or {}).get("uid")
        assert uid in KNOWN_DATASOURCES, f"{where} has datasource {uid!r}"

        assert panel.get("targets"), f"{where} has no query"
        for target in panel["targets"]:
            t_uid = (target.get("datasource") or {}).get("uid")
            assert t_uid in KNOWN_DATASOURCES, f"{where} target has datasource {t_uid!r}"


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_panels_do_not_overlap_or_stray_off_the_grid(path):
    """Grafana silently reflows overlapping panels, so a board can look nothing
    like the JSON that produced it."""
    occupied: dict[tuple[int, int], str] = {}
    for panel in load(path)["panels"]:
        pos = panel["gridPos"]
        assert pos["x"] + pos["w"] <= 24, f"{panel['title']} runs past the 24-column grid"
        for x in range(pos["x"], pos["x"] + pos["w"]):
            for y in range(pos["y"], pos["y"] + pos["h"]):
                clash = occupied.get((x, y))
                assert clash is None, f"{panel['title']} overlaps {clash}"
                occupied[(x, y)] = panel["title"]


def test_every_promql_target_queries_a_metric_the_code_records():
    """Catches the §13.2 failure mode — a dashboard citing an instrument that
    was specified but never built, which renders as a flat line rather than as
    an error, so nobody notices for months.

    Checked by converting the CODE's names into their Prometheus form, not the
    other way round: the mapping only goes one way. Dots become underscores, but
    underscores stay underscores, so `agent.offmenu_request` and a hypothetical
    `agent.offmenu.request` are the same Prometheus series.
    """
    declared = set(
        re.findall(
            r'create_(?:counter|histogram|up_down_counter)\(\s*"([^"]+)"',
            (OPS.parents[1] / "api" / "agent" / "instrumentation.py").read_text(),
        )
    )
    assert declared, "no instruments found — the extraction is broken, not the dashboards"
    prometheus_names = {name.replace(".", "_") for name in declared}

    for path in DASHBOARDS:
        for panel in load(path)["panels"]:
            for target in panel["targets"]:
                expr = target.get("expr")
                if not expr:
                    continue
                for metric in set(re.findall(r"\b((?:agent|llm)_[a-z_]+)", expr)):
                    stem = metric
                    for suffix in ("_total", "_bucket", "_sum", "_count"):
                        stem = stem.removesuffix(suffix)
                    stem = stem.removesuffix("_milliseconds")
                    assert stem in prometheus_names, (
                        f"{path.stem}: {panel['title']!r} queries {metric}, but no "
                        f"instrument maps to {stem}. Declared: {sorted(prometheus_names)}"
                    )


def test_the_datasource_provisioning_points_at_the_app_database():
    """The Shop board reads money from SQL, not from counters (§9.4)."""
    provisioned = yaml.safe_load((OPS / "provisioning" / "datasources.yaml").read_text())
    entry = provisioned["datasources"][0]

    assert entry["uid"] == "coffee-shop-db"
    assert entry["type"] == "postgres"
    assert entry["jsonData"]["database"] == "coffee_shop"


def test_sql_panels_only_read_tables_the_domain_owns():
    """A dashboard is a reader. If one of these tables is renamed, this fails
    here rather than as an empty panel nobody notices."""
    from shop.models import Base

    tables = set(Base.metadata.tables)
    for path in DASHBOARDS:
        for panel in load(path)["panels"]:
            for target in panel["targets"]:
                sql = target.get("rawSql")
                if not sql:
                    continue
                referenced = {
                    word.strip("(),")
                    for i, word in enumerate(sql.split())
                    if i and sql.split()[i - 1].upper() in {"FROM", "JOIN"}
                }
                for table in referenced:
                    assert table in tables, f"{panel['title']} reads unknown table {table!r}"
