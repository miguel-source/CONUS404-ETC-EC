"""Add gray context for the seven other Harvey-associated study watersheds."""

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT

from functools import partial

import create_hurdat2_window_figure as base
from create_sanjacinto_harvey_window_figure import ROOT, draw_actual_map


if __name__ == "__main__":
    base.main(
        map_drawer=partial(draw_actual_map, show_harvey_watersheds=True),
        output_dir=ROOT / "outputs" / "hurdat2_event_window_figure" / "harvey_all_watersheds_gray",
        stem_name="harvey_all_watersheds_event_window_definition",
    )
