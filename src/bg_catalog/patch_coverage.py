"""Validate a patch package: catalog pool vs bindings coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.bg_catalog.patch_catalog import (
    TavernMinionRecord,
    load_tavern_minions,
)
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Trigger


@dataclass(frozen=True)
class CoverageIssue:
    severity: str  # error | warning | info
    code: str
    card_id: str
    name: str
    message: str


@dataclass
class PatchCoverageReport:
    patch_dir: Path
    build: int
    patch: str
    pool_count: int
    template_count: int
    effect_binding_count: int
    issues: List[CoverageIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[CoverageIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[CoverageIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> List[CoverageIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _abilities_for(ctx: PatchContext, card_id: str):
    return ctx.effects.get(card_id, ())


def _has_trigger(ctx: PatchContext, card_id: str, trigger: Trigger) -> bool:
    return any(ab.trigger == trigger for ab in _abilities_for(ctx, card_id))


def _pool_rows(ctx: PatchContext) -> List[TavernMinionRecord]:
    catalog = ctx.patch_dir / "catalog.json"
    return [
        r
        for r in load_tavern_minions(catalog)
        if r.is_bacon_pool and not r.is_golden
    ]


def analyze_patch_coverage(
    patch_dir: Optional[str | Path] = None,
    *,
    ctx: Optional[PatchContext] = None,
) -> PatchCoverageReport:
    patch_ctx = ctx if ctx is not None else PatchContext.load(patch_dir)
    pool_rows = _pool_rows(patch_ctx)
    catalog_pool_ids = {r.id for r in pool_rows}

    report = PatchCoverageReport(
        patch_dir=patch_ctx.patch_dir,
        build=patch_ctx.build,
        patch=patch_ctx.patch,
        pool_count=len(catalog_pool_ids),
        template_count=len(patch_ctx.templates),
        effect_binding_count=len(patch_ctx.effects),
    )

    if catalog_pool_ids != set(patch_ctx.pool_ids):
        missing = sorted(catalog_pool_ids - patch_ctx.pool_ids)
        extra = sorted(patch_ctx.pool_ids - catalog_pool_ids)
        if missing:
            report.issues.append(
                CoverageIssue(
                    "error",
                    "pool_ids_missing",
                    "",
                    "",
                    f"PatchContext.pool_ids missing {len(missing)} catalog pool card(s): "
                    f"{', '.join(missing[:5])}"
                    + (" …" if len(missing) > 5 else ""),
                )
            )
        if extra:
            report.issues.append(
                CoverageIssue(
                    "error",
                    "pool_ids_extra",
                    "",
                    "",
                    f"PatchContext.pool_ids has {len(extra)} id(s) not in catalog pool: "
                    f"{', '.join(extra[:5])}"
                    + (" …" if len(extra) > 5 else ""),
                )
            )

    template_ids = set(patch_ctx.templates.keys())
    for cid in sorted(patch_ctx.effects.keys()):
        if cid not in template_ids:
            if cid.startswith("TB_BaconUps_"):
                continue
            report.issues.append(
                CoverageIssue(
                    "error",
                    "unknown_effect_key",
                    cid,
                    "",
                    f"EFFECTS[{cid!r}] has no template in patch package",
                )
            )

    for tid in sorted(patch_ctx.token_ids):
        if tid not in template_ids:
            report.issues.append(
                CoverageIssue(
                    "error",
                    "invalid_token_id",
                    tid,
                    "",
                    "TOKEN_IDS entry missing from catalog/templates",
                )
            )

    for gid in sorted(patch_ctx.golden_reward_ids):
        if gid not in template_ids:
            report.issues.append(
                CoverageIssue(
                    "error",
                    "invalid_golden_reward_id",
                    gid,
                    "",
                    "GOLDEN_REWARD_IDS entry missing from catalog/templates",
                )
            )

    for rec in pool_rows:
        cid = rec.id
        if "BATTLECRY" in rec.mechanics and not _has_trigger(patch_ctx, cid, Trigger.ON_PLACE):
            report.issues.append(
                CoverageIssue(
                    "warning",
                    "battlecry_unbound",
                    cid,
                    rec.name,
                    "catalog mechanic BATTLECRY but bindings have no Trigger.ON_PLACE",
                )
            )

        if "DEATHRATTLE" in rec.mechanics and not _has_trigger(
            patch_ctx, cid, Trigger.ON_DEATH
        ):
            report.issues.append(
                CoverageIssue(
                    "warning",
                    "deathrattle_unbound",
                    cid,
                    rec.name,
                    "catalog mechanic DEATHRATTLE but bindings have no Trigger.ON_DEATH",
                )
            )

        if cid in patch_ctx.keyword_only_pool_ids:
            continue

        if cid not in patch_ctx.effects and _looks_unimplemented(rec):
            report.issues.append(
                CoverageIssue(
                    "info",
                    "text_hook_unbound",
                    cid,
                    rec.name,
                    "pool card has non-keyword card text but no EFFECTS binding",
                )
            )

    return report


def _looks_unimplemented(rec: TavernMinionRecord) -> bool:
    if "TRIGGER_VISUAL" in rec.mechanics:
        return True
    text = (rec.text or "").strip()
    if not text:
        return False
    lower = text.lower()
    keyword_only_markers = (
        "<b>taunt</b>",
        "<b>divine shield</b>",
        "<b>poisonous</b>",
        "<b>windfury</b>",
        "<b>magnetic</b>",
        "<b>reborn</b>",
    )
    stripped = lower
    for marker in keyword_only_markers:
        stripped = stripped.replace(marker, "")
    stripped = stripped.replace("<b>", "").replace("</b>", "").strip()
    if not stripped:
        return False
    hooks = ("whenever", "each turn", "after ", "at the end", "start of combat")
    return any(h in lower for h in hooks)


def format_report(report: PatchCoverageReport) -> str:
    lines = [
        f"patch {report.patch} build {report.build}",
        f"dir   {report.patch_dir}",
        f"pool  {report.pool_count} tavern minions",
        f"templates {report.template_count}  EFFECTS keys {report.effect_binding_count}",
    ]
    for label, items in (
        ("errors", report.errors),
        ("warnings", report.warnings),
        ("info", report.infos),
    ):
        if not items:
            continue
        lines.append(f"{label} ({len(items)}):")
        for issue in items:
            prefix = issue.card_id or issue.code
            name = f" ({issue.name})" if issue.name else ""
            lines.append(f"  [{issue.code}] {prefix}{name}: {issue.message}")
    if report.ok and not report.warnings and not report.infos:
        lines.append("ok — no issues")
    elif report.ok:
        lines.append("ok — no errors")
    return "\n".join(lines)


__all__ = [
    "CoverageIssue",
    "PatchCoverageReport",
    "analyze_patch_coverage",
    "format_report",
]
