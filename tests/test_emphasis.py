"""Sizing the page by how often each bird has actually been heard.

The frame's default is unchanged - real body mass decides how big a bird is
drawn and so how central it lands. This is the other option, and most of what
matters about it is restraint: bands rather than counts, so the layout does not
move on every detection, and a breath on top of the bands, so the panel is not
asked to refresh over a bird drifting across an edge.
"""

from __future__ import annotations

import math
import time

import pytest
from PIL import Image

from fugleramme import service
from fugleramme.names import normalize
from fugleramme.picks import Picks
from fugleramme.render import collage
from fugleramme.render.collage import RANK_RAREST
from fugleramme.render.sizes import SIZE_BY_MASS, TIER_STEP, count_tiers
from fugleramme.settings import WITH_THE_BIRDS, Settings

QUIET, MIDDLE, LOUD = "Aaa aaa", "Bbb bbb", "Ccc ccc"


@pytest.fixture(autouse=True)
def _no_cached_layouts():
    collage._layouts.clear()


class _Counted:
    """Just enough of a Source for gather_entries: who was heard, how often."""

    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def species_since(self, hours: int = 24) -> list[tuple[str, int]]:
        return sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))


# --- the bands --------------------------------------------------------------


def test_the_busiest_species_defines_the_top_band():
    bands = count_tiers([("a", 200), ("b", 80), ("c", 30), ("d", 12), ("e", 1)])
    assert bands["a"] == bands["b"] == 2
    assert bands["c"] == 1
    assert bands["d"] == bands["e"] == 0


def test_the_bands_read_the_same_at_a_quiet_feeder_and_a_loud_one():
    # Relative to the busiest species, so a garden with ten times the traffic
    # does not simply put every bird in the top band.
    quiet = count_tiers([("a", 20), ("b", 3), ("c", 1)])
    loud = count_tiers([("a", 2000), ("b", 300), ("c", 100)])
    assert quiet == loud


def test_one_more_detection_never_moves_a_band():
    """The point of bands. The layout is a function of its key and the loop
    repaints when that key moves, so counts in the key would mean e-ink all day.
    """
    before = count_tiers([("a", 60), ("b", 20), ("c", 4)])
    assert count_tiers([("a", 61), ("b", 20), ("c", 4)]) == before
    assert count_tiers([("a", 60), ("b", 21), ("c", 4)]) == before
    assert count_tiers([("a", 60), ("b", 20), ("c", 5)]) == before


def test_a_species_heard_nothing_of_is_not_banded():
    assert count_tiers([]) == {}
    assert count_tiers([("a", 0)]) == {}
    assert count_tiers([("a", 4), ("b", 0)]) == {"a": 2}


# --- the weights ------------------------------------------------------------


def test_the_bands_multiply_the_mass_weights_without_moving_their_average(monkeypatch):
    """Every factor is divided by the set's geometric mean, which is what keeps
    `base` - the number that sizes the cluster to the page - from drifting."""
    names = [QUIET, MIDDLE, LOUD]
    monkeypatch.setattr(collage, "mass_of", lambda name: 40.0)
    bands = {QUIET: 0, MIDDLE: 1, LOUD: 2}

    plain = collage._size_weights(names, {})
    weighted = collage._size_weights(names, bands)

    assert plain == [pytest.approx(1.0)] * 3
    assert collage._geometric(weighted) == pytest.approx(1.0)
    assert weighted[2] / weighted[0] == pytest.approx(TIER_STEP**2)


def test_body_mass_still_has_its_say(monkeypatch):
    # A loud wren does not outgrow a quiet swan by two bands alone; the two
    # inputs multiply, they do not replace each other. They are tuned to about
    # the same strength, though, so it takes a real swan to win: three orders of
    # magnitude of mass against the full two bands.
    masses = {QUIET: 10_000.0, LOUD: 5.0}
    monkeypatch.setattr(collage, "mass_of", masses.get)
    weights = collage._size_weights([QUIET, LOUD], {QUIET: 0, LOUD: 2})
    assert weights[0] > weights[1]


# --- what it does to the page ----------------------------------------------


def _placed(names: list[str], bands: dict[str, int], size: tuple[int, int] = (1200, 900)):
    arts = [Image.new("RGBA", (100, 100), (30, 30, 30, 255)) for _ in names]
    placed, _px = collage._placements(
        (tuple(names), tuple(sorted(bands.items()))),
        arts,
        names,
        bands,
        [False] * len(names),
        *size,
        None,
        0,
        str,
    )
    return {names[p.index]: p for p in placed}


def _from_centre(p, size: tuple[int, int]) -> float:
    return math.hypot(p.at[0] + p.dim / 2 - size[0] / 2, p.at[1] + p.dim / 2 - size[1] / 2)


def test_the_loudest_bird_is_drawn_biggest_and_lands_nearest_the_centre(monkeypatch):
    monkeypatch.setattr(collage, "mass_of", lambda name: 40.0)  # mass out of the way
    size = (1200, 900)
    by_name = _placed([QUIET, MIDDLE, LOUD], {QUIET: 0, MIDDLE: 1, LOUD: 2}, size)

    assert by_name[LOUD].dim > by_name[MIDDLE].dim > by_name[QUIET].dim
    assert _from_centre(by_name[LOUD], size) == min(_from_centre(p, size) for p in by_name.values())


def test_the_quiet_birds_end_up_out_at_the_edges(monkeypatch):
    """Centrality is emergent, not assigned: the biggest goes down first on a
    centre-out spiral and the rest fill in around it. So it is the trend that
    holds, not a strict order - two birds of neighbouring bands can easily land
    the same distance out once the packed cluster is recentred on the page.
    """
    monkeypatch.setattr(collage, "mass_of", lambda name: 40.0)
    size = (1400, 1000)
    names = [f"Testus sp{i:02d}" for i in range(12)]
    bands = {name: 2 - i // 4 for i, name in enumerate(names)}  # 4 loud, 4 middling, 4 quiet

    by_name = _placed(names, bands, size)
    out = {band: [] for band in (0, 1, 2)}
    for name, p in by_name.items():
        out[bands[name]].append(_from_centre(p, size))

    assert sum(out[2]) / 4 < sum(out[1]) / 4 < sum(out[0]) / 4


def test_a_page_of_equal_birds_is_laid_out_as_if_the_bands_were_off(monkeypatch):
    monkeypatch.setattr(collage, "mass_of", lambda name: 40.0)
    names = [QUIET, MIDDLE, LOUD]
    level = _placed(names, dict.fromkeys(names, 2))
    off = _placed(names, {})
    assert {n: (p.dim, p.at) for n, p in level.items()} == {
        n: (p.dim, p.at) for n, p in off.items()
    }


# --- which birds make the page (#53) ---------------------------------------


def _garden(tmp_path, count: int, drawn: int | None = None) -> list[str]:
    """`count` species, the i-th heard i+1 times, so the busiest sort last by
    name. Only the first `drawn` of them get a plate."""
    birds = tmp_path / "classic" / "birds"
    birds.mkdir(parents=True, exist_ok=True)
    names = [f"Testus sp{i:02d}" for i in range(count)]
    for name in names[: count if drawn is None else drawn]:
        Image.new("RGBA", (20, 16), (30, 30, 30, 255)).save(birds / f"{normalize(name)}.png")
    return names


def _on_page(tmp_path, names: list[str], **kwargs) -> list[str]:
    source = _Counted({name: i + 1 for i, name in enumerate(names)})
    return collage.selected_species(source, tmp_path, "classic", **kwargs)


def test_the_frames_own_ceiling_keeps_the_most_heard_not_the_first_by_name(tmp_path):
    """MAX_BIRDS is a render budget, so which birds it spends on is a real
    choice. Cutting by name instead retired the busiest bird in the garden for
    being called Turdus rather than Anas."""
    names = _garden(tmp_path, collage.MAX_BIRDS + 5)
    kept = _on_page(tmp_path, names)

    assert set(kept) == set(names[-collage.MAX_BIRDS :])  # the busiest, not the first
    assert kept == sorted(kept)  # and still handed over in name order


def test_a_limit_keeps_the_most_heard_by_default(tmp_path):
    names = _garden(tmp_path, 20)
    assert _on_page(tmp_path, names, limit=5) == sorted(names[-5:])


def test_the_rarest_ranking_keeps_the_other_end(tmp_path):
    """What the request was for: the residents are the same every day, so the
    visitor worth seeing is the one at the bottom of the list."""
    names = _garden(tmp_path, 20)
    assert _on_page(tmp_path, names, limit=5, ranking=RANK_RAREST) == sorted(names[:5])


def test_no_limit_is_the_default_and_still_answers_to_the_frames_ceiling(tmp_path):
    names = _garden(tmp_path, collage.MAX_BIRDS + 5)
    assert len(_on_page(tmp_path, names)) == collage.MAX_BIRDS
    assert len(_on_page(tmp_path, names, limit=collage.NO_LIMIT)) == collage.MAX_BIRDS
    assert len(_on_page(tmp_path, names, limit=10)) == 10


def test_the_ranking_has_no_say_until_there_is_a_limit(tmp_path):
    """The admin greys the ranking out under "No limit", so a saved "rarest"
    must not quietly decide which forty a busy station gets."""
    names = _garden(tmp_path, collage.MAX_BIRDS + 5)
    by_default = _on_page(tmp_path, names)
    assert _on_page(tmp_path, names, ranking=RANK_RAREST) == by_default
    assert set(by_default) == set(names[-collage.MAX_BIRDS :])


def test_a_species_with_no_artwork_never_takes_a_place_from_one_that_has_it(tmp_path):
    # The loudest bird in the garden, and the style cannot draw it. Dropping it
    # after the limit rather than before would spend a place on nothing.
    names = _garden(tmp_path, 12, drawn=11)
    loudest = names[-1]

    kept = _on_page(tmp_path, names, limit=5)

    assert loudest not in kept
    assert kept == sorted(names[6:11])  # the five busiest it can actually draw


def test_the_entries_carry_the_artwork_each_selected_bird_is_wearing(tmp_path):
    names = _garden(tmp_path, 6)
    source = _Counted({name: i + 1 for i, name in enumerate(names)})

    entries = collage.gather_entries(
        source, tmp_path, "classic", Picks(tmp_path / "artwork.json"), limit=3
    )

    assert [name for name, _path in entries] == sorted(names[-3:])
    assert all(path is not None for _name, path in entries)


# --- the breath -------------------------------------------------------------


def test_the_breath_holds_a_size_change_and_nothing_else():
    now = time.monotonic()
    assert service._breathing(("same",), ("same",), now, 15) is True
    assert service._breathing(("new bird",), ("same",), now, 15) is False
    assert service._breathing(("same",), None, now, 15) is False  # nothing on the glass yet


def test_the_breath_lets_go_once_it_is_up_and_can_be_switched_off():
    now = time.monotonic()
    assert service._breathing(("same",), ("same",), now - 16 * 60, 15) is False
    assert service._breathing(("same",), ("same",), now, 0) is False


def test_by_default_a_size_change_never_spends_a_refresh_of_its_own():
    """The panel's rule (discussion #37): a refresh you notice should mean the
    frame heard a new bird. So the sizes ride along with the next one."""
    long_ago = time.monotonic() - 30 * 24 * 3600
    assert Settings().breath_minutes == WITH_THE_BIRDS
    assert service._breathing(("same",), ("same",), long_ago, WITH_THE_BIRDS) is True
    assert service._breathing(("new bird",), ("same",), long_ago, WITH_THE_BIRDS) is False


def test_sizing_by_body_mass_is_the_default():
    assert Settings().size_by == SIZE_BY_MASS
