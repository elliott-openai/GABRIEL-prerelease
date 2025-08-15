# src/gabriel/tasks/rate.py
# ════════════════════════════════════════════════════════════════════
# Robust passage-rating task with optional debug logging.
# ════════════════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional
import os
from pathlib import Path

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json, load_image_inputs, load_audio_inputs


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class RateConfig:
    attributes: Dict[str, str]
    save_dir: str = "ratings"
    file_name: str = "ratings.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 400
    n_runs: int = 1
    use_dummy: bool = False
    timeout: float = 75.0
    rating_scale: Optional[str] = None
    additional_instructions: Optional[str] = None
    modality: str = "text"
    n_attributes_per_run: int = 8
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


# ────────────────────────────
# Main rating task
# ────────────────────────────
class Rate:
    """Rate passages on specified attributes (0–100)."""


    # -----------------------------------------------------------------
    def __init__(self, cfg: RateConfig, template: Optional[PromptTemplate] = None) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    # -----------------------------------------------------------------
    # Parse raw LLM output into {attribute: float}
    # -----------------------------------------------------------------
    async def _parse(self, raw: Any, attrs: List[str]) -> Dict[str, Optional[float]]:
        obj = await safest_json(raw)
        out: Dict[str, Optional[float]] = {}
        if isinstance(obj, dict):
            for attr in attrs:
                try:
                    out[attr] = float(obj.get(attr)) if obj.get(attr) is not None else None
                except Exception:
                    out[attr] = None
            return out
        return {attr: None for attr in attrs}

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        debug: bool = False,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return ``df`` with one column per attribute rating."""

        df_proc = df.reset_index(drop=True).copy()
        values = df_proc[column_name].tolist()
        texts = [str(v) for v in values]

        base_ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_val: Dict[str, Any] = {}
        prompt_texts: Dict[str, str] = {}
        row_ids: List[str] = []

        for row, (passage, orig) in enumerate(zip(texts, values)):
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            row_ids.append(sha8)
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            id_to_val[sha8] = orig
            prompt_texts[sha8] = passage if self.cfg.modality in {"text", "entity", "web"} else ""
            base_ids.append(sha8)

        df_proc["_gid"] = row_ids

        attr_items = list(self.cfg.attributes.items())
        attr_batches: List[Dict[str, str]] = [
            dict(attr_items[i : i + self.cfg.n_attributes_per_run])
            for i in range(0, len(attr_items), self.cfg.n_attributes_per_run)
        ]

        prompts: List[str] = []
        ids: List[str] = []
        for batch_idx, batch_attrs in enumerate(attr_batches):
            for ident in base_ids:
                prompts.append(
                    self.template.render(
                        text=prompt_texts[ident],
                        attributes=batch_attrs,
                        scale=self.cfg.rating_scale,
                        additional_instructions=self.cfg.additional_instructions,
                        modality=self.cfg.modality,
                    )
                )
                ids.append(f"{ident}_batch{batch_idx}")

        prompt_images: Optional[Dict[str, List[str]]] = None
        prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None

        if self.cfg.modality == "image":
            tmp: Dict[str, List[str]] = {}
            for ident, rows in id_to_rows.items():
                imgs = load_image_inputs(values[rows[0]])
                if imgs:
                    for batch_idx in range(len(attr_batches)):
                        tmp[f"{ident}_batch{batch_idx}"] = imgs
            prompt_images = tmp or None
        elif self.cfg.modality == "audio":
            tmp_a: Dict[str, List[Dict[str, str]]] = {}
            for ident, rows in id_to_rows.items():
                auds = load_audio_inputs(values[rows[0]])
                if auds:
                    for batch_idx in range(len(attr_batches)):
                        tmp_a[f"{ident}_batch{batch_idx}"] = auds
            prompt_audio = tmp_a or None

        base_name = os.path.splitext(self.cfg.file_name)[0]
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")

        kwargs.setdefault("use_web_search", self.cfg.modality == "web")
        kwargs.setdefault("json_mode", self.cfg.modality != "audio")
        kwargs.setdefault("timeout", self.cfg.timeout)
        kwargs.setdefault("model", self.cfg.model)
        kwargs.setdefault("n_parallels", self.cfg.n_parallels)
        kwargs.setdefault("use_dummy", self.cfg.use_dummy)

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        if self.cfg.n_runs == 1:
            df_resp_all = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=prompt_images,
                prompt_audio=prompt_audio,
                save_path=csv_path,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            df_resps = [df_resp_all]
        else:
            prompts_all: List[str] = []
            ids_all: List[str] = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                prompts_all.extend(prompts)
                ids_all.extend([f"{ident}_run{run_idx}" for ident in ids])

            prompt_images_all: Optional[Dict[str, List[str]]] = None
            if prompt_images:
                prompt_images_all = {}
                for ident, imgs in prompt_images.items():
                    for run_idx in range(1, self.cfg.n_runs + 1):
                        prompt_images_all[f"{ident}_run{run_idx}"] = imgs
            prompt_audio_all: Optional[Dict[str, List[Dict[str, str]]]] = None
            if prompt_audio:
                prompt_audio_all = {}
                for ident, auds in prompt_audio.items():
                    for run_idx in range(1, self.cfg.n_runs + 1):
                        prompt_audio_all[f"{ident}_run{run_idx}"] = auds

            df_resp_all = await get_all_responses(
                prompts=prompts_all,
                identifiers=ids_all,
                prompt_images=prompt_images_all,
                prompt_audio=prompt_audio_all,
                save_path=csv_path,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )

            df_resps = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                suffix = f"_run{run_idx}"
                sub = df_resp_all[df_resp_all.Identifier.str.endswith(suffix)].copy()
                sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
                df_resps.append(sub)

        if debug:
            print("\n── raw LLM responses ──")
            for run_idx, df_resp in enumerate(df_resps, start=1):
                for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                    r = raw[0] if isinstance(raw, list) and raw else raw
                    print(f"[run {run_idx}] {ident} →\n{r}\n")
            print("────────────────────────\n")

        # parse each run and build disaggregated records
        full_records: List[Dict[str, Any]] = []
        base_attrs = list(self.cfg.attributes.keys())
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_ratings: Dict[str, Dict[str, Optional[float]]] = {
                ident: {attr: None for attr in base_attrs} for ident in base_ids
            }
            for ident_batch, raw in zip(df_resp.Identifier, df_resp.Response):
                main = raw[0] if isinstance(raw, list) and raw else raw
                base_ident, batch_part = ident_batch.rsplit("_batch", 1)
                batch_idx = int(batch_part)
                attrs = list(attr_batches[batch_idx].keys())
                parsed = await self._parse(main, attrs)
                for attr in attrs:
                    id_to_ratings[base_ident][attr] = parsed.get(attr)
            for ident in base_ids:
                parsed = id_to_ratings.get(ident, {attr: None for attr in base_attrs})
                rec = {"id": ident, "text": id_to_val[ident], "run": run_idx}
                rec.update({attr: parsed.get(attr) for attr in base_attrs})
                full_records.append(rec)

        full_df = pd.DataFrame(full_records).set_index(["id", "run"])
        disagg_path = os.path.join(self.cfg.save_dir, f"{base_name}_full_disaggregated.csv")
        full_df.to_csv(disagg_path, index_label=["id", "run"])

        # aggregate across runs
        agg_df = full_df.groupby("id")[list(self.cfg.attributes)].mean()

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        result = df_proc.merge(agg_df, left_on="_gid", right_index=True, how="left")
        result = result.drop(columns=["_gid"])
        result.to_csv(out_path, index=False)

        # keep raw response files for reference

        return result
