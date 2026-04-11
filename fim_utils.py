"""
Utilities for Fill-In-The-Middle (FIM) card tokenization.
Canonical field order and helpers for FIM block construction.
"""

from special_tokens import (
    begin_card_token,
    end_card_token,
    fim_begin_token,
    fim_end_token,
    sentinel_tokens,
)

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
