from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json, load_image_inputs, load_audio_inputs


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class ClassifyConfig:
    """Configuration for :class:`Classify`."""

    labels: Dict[str, str]  # {"label_name": "description", ...}
    save_dir: str = "classifier"
    file_name: str = "classify_responses.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 400
    n_runs: int = 1
    min_frequency: float = 0.6
    additional_instructions: str = ""
    additional_guidelines: str = ""
    use_dummy: bool = False
    timeout: float = 60.0
    modality: str = "text"
    n_attributes_per_run: int = 8
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


# ────────────────────────────
# Main Basic classifier task
# ────────────────────────────
class Classify:
    """Robust passage classifier using an LLM.

    * Accepts a list of *texts* (not a DataFrame) just like :class:`Rate`.
    * Persists/reads cached responses via the **save_path** attribute (same pattern as
      :class:`Rate`).
    """

    _FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    # -----------------------------------------------------------------
    def __init__(self, cfg: ClassifyConfig, template: Optional[PromptTemplate] = None) -> None:  # noqa: D401,E501
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package(
            "classification_prompt.jinja2"
        )

    # -----------------------------------------------------------------
    # Helpers for parsing raw model output
    # -----------------------------------------------------------------
    @staticmethod
    def _regex(raw: str, labels: List[str]) -> Dict[str, Optional[bool]]:
        out: Dict[str, Optional[bool]] = {}
        for lab in labels:
            pat = re.compile(rf'\s*"?\s*{re.escape(lab)}\s*"?\s*:\s*(true|false)', re.I | re.S)
            m = pat.search(raw)
            out[lab] = None if not m else m.group(1).lower() == "true"
        return out

    async def _parse(self, resp: Any, labels: List[str]) -> Dict[str, Optional[bool]]:
        # unwrap common response containers (list-of-one, bytes, fenced blocks)
        if isinstance(resp, list) and len(resp) == 1:
            resp = resp[0]
        if isinstance(resp, (bytes, bytearray)):
            resp = resp.decode()
        data: Optional[Any] = None
        if isinstance(resp, str):
            m = self._FENCE_RE.search(resp)
            if m:
                resp = m.group(1).strip()

            data = await safest_json(resp)
        elif isinstance(resp, dict):
            data = resp
        if isinstance(data, dict):
            norm = {
                k.strip().lower(): (
                    True
                    if str(v).strip().lower() in {"true", "yes", "1"}
                    else False
                    if str(v).strip().lower() in {"false", "no", "0"}
                    else None
                )
                for k, v in data.items()
            }
            return {lab: norm.get(lab.lower(), None) for lab in labels}

        # fallback to regex extraction
        return self._regex(str(resp), labels)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Classify items in ``df[column_name]`` and return ``df`` with label columns."""

        df_proc = df.reset_index(drop=True).copy()
        values = df_proc[column_name].tolist()

        texts = [str(v) for v in values]
        base_ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_val: Dict[str, Any] = {}
        prompt_texts: Dict[str, str] = {}

        for row, val in enumerate(texts):
            clean = " ".join(str(val).split())
            sha8 = hashlib.sha1(clean.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            id_to_val[sha8] = values[row]
            prompt_texts[sha8] = str(values[row]) if self.cfg.modality in {"text", "entity", "web"} else ""
            base_ids.append(sha8)

        label_items = list(self.cfg.labels.items())
        label_batches: List[Dict[str, str]] = [
            dict(label_items[i : i + self.cfg.n_attributes_per_run])
            for i in range(0, len(label_items), self.cfg.n_attributes_per_run)
        ]

        prompts: List[str] = []
        ids: List[str] = []
        for batch_idx, batch_labels in enumerate(label_batches):
            for ident in base_ids:
                prompts.append(
                    self.template.render(
                        text=prompt_texts[ident],
                        attributes=batch_labels,
                        additional_instructions=self.cfg.additional_instructions,
                        additional_guidelines=self.cfg.additional_guidelines,
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
                    for batch_idx in range(len(label_batches)):
                        tmp[f"{ident}_batch{batch_idx}"] = imgs
            prompt_images = tmp or None
        elif self.cfg.modality == "audio":
            tmp_a: Dict[str, List[Dict[str, str]]] = {}
            for ident, rows in id_to_rows.items():
                auds = load_audio_inputs(values[rows[0]])
                if auds:
                    for batch_idx in range(len(label_batches)):
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
                print_example_prompt=True,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")
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
                print_example_prompt=True,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")

            df_resps = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                suffix = f"_run{run_idx}"
                sub = df_resp_all[df_resp_all.Identifier.str.endswith(suffix)].copy()
                sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
                df_resps.append(sub)

        # parse each run and construct disaggregated records
        full_records: List[Dict[str, Any]] = []
        total_orphans = 0
        all_labels = list(self.cfg.labels.keys())
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_labels: Dict[str, Dict[str, Optional[bool]]] = {
                ident: {lab: None for lab in all_labels} for ident in base_ids
            }
            orphans = 0
            for ident_batch, raw in zip(df_resp.Identifier, df_resp.Response):
                if "_batch" not in ident_batch:
                    continue
                base_ident, batch_part = ident_batch.rsplit("_batch", 1)
                if base_ident not in id_to_rows:
                    orphans += 1
                    continue
                batch_idx = int(batch_part)
                labs = list(label_batches[batch_idx].keys())
                parsed = await self._parse(raw, labs)
                for lab in labs:
                    id_to_labels[base_ident][lab] = parsed.get(lab)
            total_orphans += orphans
            for ident in base_ids:
                parsed = id_to_labels.get(ident, {lab: None for lab in all_labels})
                rec = {"text": id_to_val[ident], "run": run_idx}
                rec.update({lab: parsed.get(lab) for lab in all_labels})
                full_records.append(rec)

        if total_orphans:
            print(
                f"[Classify] WARNING: {total_orphans} response(s) had no matching passage this run."
            )

        full_df = pd.DataFrame(full_records).set_index(["text", "run"])
        disagg_path = os.path.join(self.cfg.save_dir, f"{base_name}_full_disaggregated.csv")
        full_df.to_csv(disagg_path, index_label=["text", "run"])

        # aggregate across runs using a minimum frequency threshold
        def _min_freq(s: pd.Series) -> Optional[bool]:
            if s.notna().sum() == 0:
                return None
            true_count = s.fillna(False).sum()
            prop = true_count / self.cfg.n_runs
            return prop >= self.cfg.min_frequency

        agg_df = pd.DataFrame({lab: full_df[lab].groupby("text").apply(_min_freq) for lab in self.cfg.labels})

        filled = agg_df.dropna(how="all").shape[0]
        print(f"[Classify] Filled {filled}/{len(agg_df)} unique texts.")

        total = len(agg_df)
        print("\n=== Label coverage (non-null) ===")
        for lab in self.cfg.labels:
            n = agg_df[lab].notna().sum()
            print(f"{lab:<55s}: {n / total:6.2%} ({n}/{total})")
        print("=================================\n")

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        result = df_proc.merge(agg_df, left_on=column_name, right_index=True, how="left")
        result.to_csv(out_path, index=False)

        # keep raw response files for reference

        return result

