"""
Utilities for Fill-In-The-Middle (FIM) card tokenization.
Canonical field order and helpers for FIM block construction.
"""

import sys
from dataclasses import dataclass

from special_tokens import (
    begin_card_token,
    end_card_token,
    end_mana_cost_token,
    end_name_token,
    end_oracle_text_token,
    end_type_line_token,
    fim_begin_token,
    fim_end_token,
    sentinel_tokens,
)

from tokenizers.tokenize_mana_cost import tokenize_mana_cost
from tokenizers.tokenize_name import tokenize_name
from tokenizers.tokenize_oracle_text import tokenize_oracle_text
from tokenizers.tokenize_type_line import tokenize_type_line

# Fixed order for all card fields (standard and FIM mode).
# Optional fields (power, toughness, loyalty) are included only when present on the card.
CANONICAL_FIELD_ORDER = [
    "name",
    "mana_cost",
    "type_line",
    "oracle_text",
    "rarity",
    "set",
    "release_year",
    "power",
    "toughness",
    "loyalty",
]


# Map canonical field name to Card attribute name (for inference partial cards).
FIELD_TO_ATTR = {
    "name": "name",
    "mana_cost": "mana_cost",
    "type_line": "type_line",
    "oracle_text": "oracle_text",
    "rarity": "rarity",
    "set": "set_code",
    "release_year": "release_year",
    "power": "power",
    "toughness": "toughness",
    "loyalty": "loyalty",
}


def _card_has_field(card, field_name):
    """Return True if the card has a value for the given canonical field name."""
    attr = FIELD_TO_ATTR.get(field_name)
    return attr is not None and getattr(card, attr, None) is not None


def get_canonical_fields_for_card(card):
    """
    Return the list of fields for this card in canonical order.
    Only includes fields that are present (optional fields omitted when None).
    Matches the same availability logic as get_available_fields in tokenize_card_file_permuted.

    Args:
        card: Card object

    Returns:
        List of field names in CANONICAL_FIELD_ORDER, subset for this card
    """
    # Required fields (always present from Card.from_json)
    available = {"name", "mana_cost", "type_line", "release_year", "rarity", "set", "oracle_text"}
    if card.type_line is not None:
        if 'Creature' in card.type_line or 'Vehicle' in card.type_line or 'Spacecraft' in card.type_line:
            if card.power is not None:
                available.add("power")
            if card.toughness is not None:
                available.add("toughness")
        if 'Planeswalker' in card.type_line:
            if card.loyalty is not None:
                available.add("loyalty")
    return [f for f in CANONICAL_FIELD_ORDER if f in available]


def get_inference_fields_for_card(card):
    """
    Return canonical fields that should be considered by inference gap-planning.
    Unlike get_canonical_fields_for_card, this includes optional fields that are
    implied by type_line even when currently missing (e.g. creature power/toughness).
    """
    fields = {"name", "mana_cost", "type_line", "release_year", "rarity", "set", "oracle_text"}
    type_line = card.type_line or ""
    if "Creature" in type_line or "Vehicle" in type_line or "Spacecraft" in type_line:
        fields.add("power")
        fields.add("toughness")
    if "Planeswalker" in type_line:
        fields.add("loyalty")
    return [f for f in CANONICAL_FIELD_ORDER if f in fields]


def _compute_runs(canonical_fields, mask_set):
    """
    Compute runs of consecutive masked fields.
    Returns a list of lists; each inner list is a run of consecutive masked field names.
    """
    runs = []
    current_run = []
    for f in canonical_fields:
        if f in mask_set:
            current_run.append(f)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)
    return runs


def build_fim_block(card, mask_set):
    """
    Build a single FIM token block for a card with the given fields masked.

    Format: <card> <FIM> [body with sentinels] </FIM> <sentinel_0> [content of run 0] ... </card>
    Sentinel content includes field tags (e.g. <mana_cost> R </mana_cost>).

    Args:
        card: Card object
        mask_set: Set (or sequence) of field names to mask; must be subset of card's canonical fields.
                  Use 1 < len(mask_set) < len(canonical_fields).

    Returns:
        List of token strings for the FIM block.

    Raises:
        ValueError: if mask_set is invalid or we need more sentinels than defined.
    """
    canonical_fields = get_canonical_fields_for_card(card)
    mask_set = set(mask_set)
    if not mask_set.issubset(set(canonical_fields)):
        raise ValueError("mask_set must be a subset of the card's canonical fields")
    if len(mask_set) >= len(canonical_fields) or len(mask_set) < 1:
        raise ValueError("Need 1 < len(mask_set) < len(canonical_fields)")

    runs = _compute_runs(canonical_fields, mask_set)
    if len(runs) > len(sentinel_tokens):
        raise ValueError(
            f"Too many runs ({len(runs)}); only {len(sentinel_tokens)} sentinels defined. "
            "Use a mask pattern with fewer contiguous runs."
        )

    # Build body: walk canonical order; emit tokens for unmasked, one sentinel per run for masked
    body = []
    run_index = 0
    in_run = False
    for f in canonical_fields:
        if f in mask_set:
            if not in_run:
                body.append(sentinel_tokens[run_index])
                run_index += 1
                in_run = True
            # else: same run, no extra sentinel
        else:
            in_run = False
            body.extend(card.generate_tokens([f]))

    # Build tail: <sentinel_i> [content of run i] for each run
    tail = []
    for i, run_fields in enumerate(runs):
        tail.append(sentinel_tokens[i])
        tail.extend(card.generate_tokens(run_fields))

    return (
        [begin_card_token, fim_begin_token]
        + body
        + [fim_end_token]
        + tail
        + [end_card_token]
    )


def build_fim_prompt_for_inference(partial_card):
    """
    Build the FIM prompt token list for inference from a partial card.
    Missing fields (None) are represented by sentinels in the body; the prompt
    ends with </FIM><sentinel_0> so the model generates the first gap's content.

    Args:
        partial_card: Card object with some fields set (others None)

    Returns:
        (prompt_tokens, runs)
        - prompt_tokens: list of token strings to feed as context (ends with <sentinel_0>)
        - runs: list of lists; runs[i] is the list of field names for the i-th gap,
                so the model will generate content for runs[0], then <sentinel_1>, then runs[1], etc.
        If no fields are missing, returns (full_card_tokens, []).
    """
    missing_set = {
        f for f in CANONICAL_FIELD_ORDER
        if not _card_has_field(partial_card, f)
    }
    if not missing_set:
        # No missing fields: return full card in canonical order (only present fields)
        fields = get_canonical_fields_for_card(partial_card)
        tokens = (
            [begin_card_token]
            + partial_card.generate_tokens(fields)
            + [end_card_token]
        )
        return (tokens, [])

    runs = _compute_runs(CANONICAL_FIELD_ORDER, missing_set)
    if len(runs) > len(sentinel_tokens):
        runs = runs[: len(sentinel_tokens)]
        missing_set = {f for run in runs for f in run}

    body = []
    run_index = 0
    in_run = False
    for f in CANONICAL_FIELD_ORDER:
        if f in missing_set:
            if not in_run:
                body.append(sentinel_tokens[run_index])
                run_index += 1
                in_run = True
        else:
            in_run = False
            body.extend(partial_card.generate_tokens([f]))

    prompt_tokens = (
        [begin_card_token, fim_begin_token]
        + body
        + [fim_end_token, sentinel_tokens[0]]
    )
    return (prompt_tokens, runs)


# Fields where user-facing "..." can open a real token prefix (begin tag + content, no end tag).
_FIELDS_WITH_OPEN_PREFIX = frozenset(
    {"name", "mana_cost", "type_line", "oracle_text"}
)


def parse_field_value_for_inference(field_name, raw):
    """
    Interpret optional inference ellipsis in user-provided field strings.

    Valid: exactly "...", or any string that ends with "..." whose prefix does not
    contain "..." (single trailing completion marker).

    Returns:
        (stored_value, wants_partial_completion, malformed_treat_as_literal)
        - stored_value: str to store on Card (prefix after stripping marker), or None
        - wants_partial_completion: True if model should continue inside this field's block
        - malformed_treat_as_literal: True if "..." rules were violated; caller should keep raw string
    """
    if raw is None:
        return (None, False, False)
    if "..." not in raw:
        return (raw, False, False)
    if raw == "...":
        if field_name not in _FIELDS_WITH_OPEN_PREFIX:
            return (None, False, False)
        # Store None on Card; partial_fields marks the open block to complete
        return (None, True, False)
    if raw.endswith("..."):
        prefix = raw[:-3]
        if "..." in prefix:
            return (raw, False, True)
        if field_name not in _FIELDS_WITH_OPEN_PREFIX:
            return (raw, False, True)
        return (prefix, True, False)
    return (raw, False, True)


def normalize_inference_string_field(field_name, raw, warn_fn=None):
    """
    Apply parse_field_value_for_inference; on malformed, optionally warn and return literal raw.
    """
    stored, partial, malformed = parse_field_value_for_inference(field_name, raw)
    if malformed and warn_fn is not None:
        warn_fn(
            f"Inference: field {field_name!r} has invalid use of '...' (only a lone '...' or "
            f"a single trailing '...' after text with no '...' in the prefix is allowed). "
            f"Treating value as literal."
        )
    if malformed:
        return (raw, False)
    return (stored, partial)


def _field_applies_in_inference_order(card, field_name):
    """Whether this canonical field participates in ordering for this card."""
    return field_name in get_inference_fields_for_card(card)


def _field_is_gap(card, field_name, partial_fields):
    if not _field_applies_in_inference_order(card, field_name):
        return False
    if field_name in partial_fields:
        return True
    return not _card_has_field(card, field_name)


def find_leftmost_inference_gap(card, partial_fields):
    """
    First canonical field that still needs model work (missing or partial ellipsis).
    Returns field name or None if none.
    """
    for f in CANONICAL_FIELD_ORDER:
        if _field_is_gap(card, f, partial_fields):
            return f
    return None


def _closed_field_tokens(card, field_name):
    return card.generate_tokens([field_name])


def open_field_prefix_tokens(card, field_name):
    """Tokens for an incomplete field block: begin tag through content, no end tag."""


    if field_name == "name":
        toks = tokenize_name(card.name or "")
        return toks[:-1] if toks and toks[-1] == end_name_token else toks
    if field_name == "mana_cost":
        toks = tokenize_mana_cost(card.mana_cost or "")
        return toks[:-1] if toks and toks[-1] == end_mana_cost_token else toks
    if field_name == "type_line":
        toks = tokenize_type_line(card.type_line or "")
        return toks[:-1] if toks and toks[-1] == end_type_line_token else toks
    if field_name == "oracle_text":
        toks = tokenize_oracle_text(
            card.oracle_text or "",
            card.name,
            card.type_line,
            card.related_card_names,
        )
        return toks[:-1] if toks and toks[-1] == end_oracle_text_token else toks
    raise ValueError(f"Open prefix not supported for field {field_name!r}")


FIELD_END_TAG = {
    "name": end_name_token,
    "mana_cost": end_mana_cost_token,
    "type_line": end_type_line_token,
    "oracle_text": end_oracle_text_token,
}


@dataclass
class InferencePromptPlan:
    prompt_tokens: list
    parse_mode: str  # "fim_sentinel_tail" | "continue_open_field" | "complete_card"
    runs: list  # for FIM: list of lists of field names; empty if not FIM
    open_field: str | None  # for continue_open_field
    end_tag: str | None


def build_inference_prompt_for_leftmost_gap(card, partial_fields):
    """
    One generate+parse round: context ends where the model should continue.

    Returns InferencePromptPlan with parse_mode:
    - complete_card: no gaps; full <card>...</card> (no generation required for filling)
    - continue_open_field: open_field + end_tag set
    - fim_sentinel_tail: runs possibly non-empty
    """
    gap = find_leftmost_inference_gap(card, partial_fields)
    if gap is None:
        fields = get_canonical_fields_for_card(card)
        tokens = (
            [begin_card_token]
            + card.generate_tokens(fields)
            + [end_card_token]
        )
        return InferencePromptPlan(
            prompt_tokens=tokens,
            parse_mode="complete_card",
            runs=[],
            open_field=None,
            end_tag=None,
        )

    # Closed complete blocks for all applicable fields strictly before `gap`
    prefix_tokens = [begin_card_token]
    for f in CANONICAL_FIELD_ORDER:
        if f == gap:
            break
        if not _field_applies_in_inference_order(card, f):
            continue
        if _field_is_gap(card, f, partial_fields):
            # Should not happen if gap is leftmost
            continue
        prefix_tokens.extend(_closed_field_tokens(card, f))

    if gap in partial_fields:
        prefix_tokens.extend(open_field_prefix_tokens(card, gap))
        return InferencePromptPlan(
            prompt_tokens=prefix_tokens,
            parse_mode="continue_open_field",
            runs=[],
            open_field=gap,
            end_tag=FIELD_END_TAG[gap],
        )

    # First gap is fully missing: scoped FIM body from `gap` onward
    idx_gap = CANONICAL_FIELD_ORDER.index(gap)
    missing_run = []
    for j in range(idx_gap, len(CANONICAL_FIELD_ORDER)):
        f = CANONICAL_FIELD_ORDER[j]
        if not _field_applies_in_inference_order(card, f):
            continue
        if not _card_has_field(card, f) and f not in partial_fields:
            missing_run.append(f)
        else:
            break

    if not missing_run:
        raise ValueError(f"Inference gap at {gap!r} but no missing run computed")

    runs = _compute_runs(CANONICAL_FIELD_ORDER, set(missing_run))
    first_run = runs[0] if runs else missing_run
    runs = [first_run]

    if len(runs[0]) > len(sentinel_tokens):
        print(
            f"Warning: inference missing run needs more than {len(sentinel_tokens)} sentinels; truncating.",
            file=sys.stderr,
        )
        runs = [runs[0][: len(sentinel_tokens)]]

    missing_set = set(runs[0])
    body = []
    run_index = 0
    in_run = False
    seen_end_of_missing_run = False
    for f in CANONICAL_FIELD_ORDER[idx_gap:]:
        if not _field_applies_in_inference_order(card, f):
            continue
        if f in missing_set:
            if seen_end_of_missing_run:
                break
            if not in_run:
                body.append(sentinel_tokens[run_index])
                run_index += 1
                in_run = True
        else:
            if in_run:
                seen_end_of_missing_run = True
                in_run = False
            if seen_end_of_missing_run:
                if _field_is_gap(card, f, partial_fields):
                    break
                body.extend(_closed_field_tokens(card, f))

    prompt_tokens = (
        prefix_tokens
        + [fim_begin_token]
        + body
        + [fim_end_token, sentinel_tokens[0]]
    )
    return InferencePromptPlan(
        prompt_tokens=prompt_tokens,
        parse_mode="fim_sentinel_tail",
        runs=runs,
        open_field=None,
        end_tag=None,
    )
