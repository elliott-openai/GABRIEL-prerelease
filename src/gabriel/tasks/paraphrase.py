from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses

# Import classifier utilities for recursive validation.  Importing from
# ``gabriel.tasks.classify`` does not introduce a circular dependency
# because the classifier does not import the paraphrasing task.
from ..tasks.classify import Classify, ClassifyConfig


@dataclass
class ParaphraseConfig:
    """Configuration for :class:`Paraphrase`."""

    # Instruction passed to the paraphrase prompt.  This should describe
    # how the original passage should be rewritten.
    instructions: str
    # Optional name for the revised column in the output DataFrame.  If
    # unspecified, ``f"{column_name}_revised"`` will be used.
    revised_column_name: Optional[str] = None
    # Number of revisions to generate per passage.  When greater than 1,
    # additional columns will be appended to the output DataFrame with
    # suffixes ``_1``, ``_2``, etc.
    n_revisions: int = 1
    # Directory where all paraphrase responses and intermediate files are
    # persisted.
    save_dir: str = "paraphrase"
    # Base file name for the raw paraphrase responses.  Cleaned and
    # validated responses will be written using the same stem with
    # ``_cleaned.csv`` appended.
    file_name: str = "paraphrase_responses.csv"
    # OpenAI model used for both paraphrasing and classification.  The
    # default matches the existing behaviour of GABRIEL.
    model: str = "gpt-5-mini"
    # When true, the model will be asked to output JSON only.
    json_mode: bool = False
    # If true, the underlying helper will use web search augmentation.
    use_web_search: bool = False
    # Maximum number of parallel requests that will be sent to the
    # underlying API.  Note that classification and paraphrasing share
    # this value for simplicity.
    n_parallels: int = 400
    # Use dummy responses instead of real API calls.  Exposed here
    # primarily for testing.
    use_dummy: bool = False
    # Optional reasoning effort passed through to the LLM helper.
    reasoning_effort: Optional[str] = None
    # Optional reasoning summary passed through to the LLM helper.
    reasoning_summary: Optional[str] = None
    # New feature: enable recursive validation of paraphrases.  When set
    # to ``True``, every generated paraphrase will be fed into a
    # classifier to verify that the instructions were followed.  Any
    # paraphrases that fail this check will be regenerated on the
    # original text until they pass.
    recursive_validation: bool = False
    # Number of candidate paraphrases to generate per failing passage in
    # subsequent validation rounds.  This parameter is ignored on the
    # initial round because only one paraphrase per passage is generated
    # initially.
    n_validation_candidates: int = 10


class Paraphrase:
    """Paraphrase text columns in a DataFrame."""

    def __init__(self, cfg: ParaphraseConfig, template: Optional[PromptTemplate] = None) -> None:
        self.cfg = cfg
        # Load the paraphrasing prompt from package data if a custom
        # template is not provided.
        self.template = template or PromptTemplate.from_package("paraphrase_prompt.jinja2")
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Paraphrase ``df[column_name]`` and return a DataFrame with revisions.

        This method orchestrates prompt construction, asynchronous API
        calls, optional recursive validation using the classifier, and
        persistence of results.  The output DataFrame preserves the
        order of the input and appends one or more revised columns.
        """
        # Ensure row indices are contiguous so we can map responses back
        # deterministically.  A copy is created to avoid mutating the
        # caller's DataFrame.
        df_proc = df.reset_index(drop=True).copy()
        # Convert the target column into a list of strings.  We coerce
        # values to strings so that non-string columns (e.g. numbers) are
        # handled gracefully.
        texts: List[str] = df_proc[column_name].astype(str).tolist()
        # Determine the base name for the revised column(s).
        base_col = self.cfg.revised_column_name or f"{column_name}_revised"
        # Determine how many paraphrases to produce per passage.  A value
        # less than 1 defaults to a single revision to align with
        # existing behaviour.
        n = self.cfg.n_revisions if self.cfg.n_revisions and self.cfg.n_revisions > 0 else 1

        # Build prompts and identifiers for the initial paraphrasing
        # request.  We use one-based revision numbers in the identifier
        # because downstream code and filenames historically follow this
        # convention.
        prompts: List[str] = []
        identifiers: List[str] = []
        for idx, text in enumerate(texts):
            for j in range(1, n + 1):
                prompts.append(
                    self.template.render(text=text, instructions=self.cfg.instructions)
                )
                identifiers.append(f"row_{idx}_rev{j}")

        # Persist raw paraphrase responses to disk.  This enables
        # inspection and debugging of intermediate results.  The helper
        # will append to an existing file unless ``reset_files`` is
        # supplied.
        save_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        kwargs.setdefault("model", self.cfg.model)
        kwargs.setdefault("json_mode", self.cfg.json_mode)
        kwargs.setdefault("use_web_search", self.cfg.use_web_search)
        kwargs.setdefault("n_parallels", self.cfg.n_parallels)
        kwargs.setdefault("use_dummy", self.cfg.use_dummy)
        resp_df = await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            save_path=save_path,
            reset_files=reset_files,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            **kwargs,
        )

        # Create a mapping from (row index, zero-based revision index) to
        # the paraphrased text.  We unwrap list-of-one responses that
        # arise from JSON-mode for convenience.
        resp_map: Dict[Tuple[int, int], str] = {}
        for ident, resp in zip(resp_df["Identifier"], resp_df["Response"]):
            main = resp[0] if isinstance(resp, list) and resp else resp
            m = re.match(r"row_(\d+)_rev(\d+)", ident)
            if m:
                row = int(m.group(1))
                # Convert one-based revision number to zero-based index.
                rev = int(m.group(2)) - 1
                resp_map[(row, rev)] = main

        # If recursive validation is enabled, run the validation loop.  This
        # call mutates ``resp_map`` in place to ensure that only
        # validated paraphrases remain.
        if self.cfg.recursive_validation:
            await self._recursive_validate(texts, resp_map, reset_files)

        # Assemble the final columns.  When multiple revisions are
        # requested, each revision will occupy its own column with a
        # numeric suffix.
        col_names = [base_col] if n == 1 else [f"{base_col}_{i}" for i in range(1, n + 1)]
        for j, col in enumerate(col_names):
            df_proc[col] = [resp_map.get((i, j), "") for i in range(len(df_proc))]

        # Persist the cleaned and validated DataFrame to disk.  This file
        # excludes metadata columns such as ``Identifier`` or ``Response``
        # and is intended for downstream analysis.
        out_path = os.path.join(
            self.cfg.save_dir,
            f"{os.path.splitext(self.cfg.file_name)[0]}_cleaned.csv",
        )
        df_proc.to_csv(out_path, index=False)
        return df_proc

    async def _recursive_validate(
        self,
        original_texts: List[str],
        resp_map: Dict[Tuple[int, int], str],
        reset_files: bool = False,
    ) -> None:
        """
        Validate each paraphrased text against the original instruction using
        a classifier.  Any paraphrases that do not strictly follow the
        instruction are regenerated on the original text until they pass.

        The first validation round checks the paraphrases that have already
        been generated.  On subsequent rounds, new paraphrases are
        generated for every failing passage; up to
        ``n_validation_candidates`` candidates are produced per passage.
        The classifier is then invoked once on all candidates to
        determine whether any candidate meets the instruction.  If none
        of the candidates pass, the first candidate is retained and the
        passage is scheduled for another round.  The loop terminates
        once all paraphrases have passed validation or when no new
        paraphrases can be generated.
        """
        # Initialise the list of keys to validate; keys are tuples of the
        # form (row index, zero-based revision index).
        to_check: List[Tuple[int, int]] = list(resp_map.keys())
        round_number = 1

        # Configure the classifier used for validation.  We reuse the
        # paraphrasing model and parallelism settings to maintain
        # consistency across tasks.  A dedicated validation directory is
        # used to avoid polluting the main paraphrase outputs.
        validation_dir = os.path.join(self.cfg.save_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        labels = {
            "instructions_followed": (
                "True if the modified text fully and faithfully follows the instruction "
                "given to transform the original text; false otherwise. "
                "Only True if the modification instructions were totally complied with in transforming the original text into the modified text."
            )
        }
        classify_cfg = ClassifyConfig(
            labels=labels,
            save_dir=validation_dir,
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            n_runs=1,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
        )
        classifier = Classify(classify_cfg)

        while to_check:
            # Build classification prompts for the current paraphrases that
            # require validation.  Each prompt contains the instruction,
            # original text and paraphrased text in a clearly labelled
            # structure so the classifier can focus on adherence.
            cls_texts: List[str] = []
            cls_keys: List[Tuple[int, int]] = []
            for key in to_check:
                row_idx, rev_idx = key
                original = original_texts[row_idx]
                paraphrased = resp_map.get(key, "")
                cls_prompt = (
                    "INSTRUCTIONS:\n"
                    f"{self.cfg.instructions.strip()}\n\n"
                    "BEGIN ORIGINAL TEXT:\n"
                    f"{original.strip()}\n"
                    "END ORIGINAL TEXT\n\n"
                    "BEGIN MODIFIED TEXT:\n"
                    f"{paraphrased.strip()}\n"
                    "END MODIFIED TEXT\n\n"
                    "Does the modified text fully and faithfully apply the instruction "
                    "to the original text to create the modified text, without introducing any deviations or omissions in following the instructions?"
                    "Again, the modification instructions that need to be validated are: "
                    f"{self.cfg.instructions.strip()}\n\n"
                )
                cls_texts.append(cls_prompt)
                cls_keys.append(key)

            cls_df = pd.DataFrame({"text": cls_texts})
            # Invoke the classifier on all prompts in this round.  The
            # resulting DataFrame includes a boolean column named
            # ``instructions_followed`` for our single label.
            res_df = await classifier.run(
                cls_df, column_name="text", reset_files=reset_files
            )
            # Identify which paraphrases failed validation.  Any value
            # other than True (including False and None) counts as a
            # failure.
            failed_keys: List[Tuple[int, int]] = []
            for idx, key in enumerate(cls_keys):
                flag = res_df.loc[idx, "instructions_followed"]
                if not bool(flag):
                    failed_keys.append(key)

            # If nothing failed, we are done.
            if not failed_keys:
                break

            # Determine how many new paraphrases to generate per failing
            # passage for the next round.
            num_candidates = self.cfg.n_validation_candidates

            # Construct new paraphrase prompts for every failing passage.
            new_prompts: List[str] = []
            new_identifiers: List[str] = []
            for key in failed_keys:
                row_idx, rev_idx = key
                original = original_texts[row_idx]
                for cand_idx in range(num_candidates):
                    new_prompts.append(
                        self.template.render(
                            text=original, instructions=self.cfg.instructions
                        )
                    )
                    # The identifier encodes the row, revision, round and candidate
                    # number so we can recover which response belongs to which
                    # passage after the API call completes.  Revision numbers are
                    # stored one-based in the identifier.
                    new_identifiers.append(
                        f"row_{row_idx}_rev{rev_idx + 1}_round{round_number + 1}_cand{cand_idx}"
                    )

            # If no new prompts were constructed (which should not happen
            # unless ``failed_keys`` is empty), exit the loop to avoid an
            # infinite loop.
            if not new_prompts:
                break

            # Request new paraphrases for all failing passages.
            tmp_save_path = os.path.join(
                self.cfg.save_dir,
                f"{os.path.splitext(self.cfg.file_name)[0]}_round{round_number + 1}.csv",
            )
            new_resp_df = await get_all_responses(
                prompts=new_prompts,
                identifiers=new_identifiers,
                save_path=tmp_save_path,
                model=self.cfg.model,
                json_mode=self.cfg.json_mode,
                use_web_search=self.cfg.use_web_search,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
            )

            # Organise responses by (row, revision) so we can classify
            # candidates in bulk.
            candidate_map: Dict[Tuple[int, int], List[str]] = {
                key: [] for key in failed_keys
            }
            for ident, resp in zip(new_resp_df["Identifier"], new_resp_df["Response"]):
                text = resp[0] if isinstance(resp, list) and resp else resp
                m = re.match(r"row_(\d+)_rev(\d+)_round\d+_cand(\d+)", ident)
                if m:
                    row_i = int(m.group(1))
                    rev_i = int(m.group(2)) - 1
                    candidate_map.setdefault((row_i, rev_i), []).append(text)

            # Build classification prompts for every candidate across all
            # failing passages.  We record the triplet (row, revision,
            # candidate index) so that we can map back classification
            # results later.
            cand_cls_texts: List[str] = []
            cand_keys: List[Tuple[int, int, int]] = []
            for key in failed_keys:
                row_idx, rev_idx = key
                original = original_texts[row_idx]
                candidates = candidate_map.get(key, [])
                for cand_index, cand_text in enumerate(candidates):
                    cls_prompt = (
                        "INSTRUCTIONS:\n"
                        f"{self.cfg.instructions.strip()}\n\n"
                        "BEGIN ORIGINAL TEXT:\n"
                        f"{original.strip()}\n"
                        "END ORIGINAL TEXT\n\n"
                        "BEGIN MODIFIED TEXT:\n"
                        f"{paraphrased.strip()}\n"
                        "END MODIFIED TEXT\n\n"
                        "Does the modified text fully and faithfully apply the instruction "
                        "to the original text to create the modified text, without introducing any deviations or omissions in following the instructions?"
                        "Again, the modification instructions that need to be validated are: "
                        f"{self.cfg.instructions.strip()}\n\n"
                    )
                    cand_cls_texts.append(cls_prompt)
                    cand_keys.append((row_idx, rev_idx, cand_index))

            # Run the classifier on all candidate prompts simultaneously.
            # If there are no candidates (which should not happen), we
            # construct an empty DataFrame so indexing operations do not
            # raise errors.
            if cand_cls_texts:
                cand_df = pd.DataFrame({"text": cand_cls_texts})
                cand_res_df = await classifier.run(
                    cand_df, column_name="text", reset_files=reset_files
                )
            else:
                cand_res_df = pd.DataFrame()

            # Build a lookup from (row, revision) to the list of boolean
            # classification results corresponding to each candidate.
            cand_results_map: Dict[Tuple[int, int], List[bool]] = {
                key: [False] * len(candidate_map.get(key, [])) for key in failed_keys
            }
            for idx, (row_idx, rev_idx, cand_index) in enumerate(cand_keys):
                if not cand_res_df.empty:
                    flag = cand_res_df.loc[idx, "instructions_followed"]
                else:
                    flag = None
                cand_results_map[(row_idx, rev_idx)][cand_index] = bool(flag)

            # Determine which passages still need another round of
            # paraphrasing.  For each failing passage, select the first
            # candidate that passed classification; if none passed, use
            # the first candidate and mark the passage for another round.
            next_to_check: List[Tuple[int, int]] = []
            for key in failed_keys:
                row_idx, rev_idx = key
                candidates = candidate_map.get(key, [])
                results = cand_results_map.get(key, [])
                chosen_text: Optional[str] = None
                passed_flag = False
                for cand_text, passed in zip(candidates, results):
                    if passed:
                        chosen_text = cand_text
                        passed_flag = True
                        break
                # If no candidate passed, fall back to the first candidate
                # if available.
                if chosen_text is None and candidates:
                    chosen_text = candidates[0]
                    passed_flag = False
                # Update the paraphrase mapping with the chosen text if
                # available.
                if chosen_text is not None:
                    resp_map[(row_idx, rev_idx)] = chosen_text
                    # Schedule another round if the chosen candidate did
                    # not pass validation.
                    if not passed_flag:
                        next_to_check.append(key)
                else:
                    # If no candidates were generated, keep the key for
                    # another round.
                    next_to_check.append(key)

            # Prepare for the next validation round.
            to_check = next_to_check
            round_number += 1
