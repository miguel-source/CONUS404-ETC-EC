"""We keep data locations configurable and independent of personal computers."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get('TC_PRECIP_PROJECT', REPO_ROOT / 'data')).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get('TC_PRECIP_RESULTS', REPO_ROOT / 'results')).expanduser().resolve()
METADATA_ROOT = REPO_ROOT / 'data' / 'metadata'
WATERSHEDS_FILE = Path(os.environ.get('TC_PRECIP_WATERSHEDS', PROJECT_ROOT / 'boundaries' / 'study_watersheds.gpkg')).expanduser().resolve()
STATES_FILE = Path(os.environ.get('TC_PRECIP_STATES', PROJECT_ROOT / 'boundaries' / 'conus_states.gpkg')).expanduser().resolve()
DAILY_INPUT = Path(os.environ.get('TC_PRECIP_DAILY', PROJECT_ROOT / 'ETC_StormEvents' / 'Gridded_PERSIANN' / 'hurricane_daily_all4_persiann_grid.nc')).expanduser().resolve()
EC_DIRECTORY = RESULTS_ROOT / 'Hurricane_gridded_EC_results'
EC_INPUT = Path(os.environ.get('TC_PRECIP_EC', EC_DIRECTORY / 'gridded_extended_collocation_all_windows.nc')).expanduser().resolve()
LOSO_INPUT = EC_INPUT.with_name('gridded_extended_collocation_all_windows_leave_one_storm_out.nc')
AGREEMENT_DIRECTORY = RESULTS_ROOT / 'Hurricane_gridded_agreement_results'
AGREEMENT_INPUT = Path(os.environ.get('TC_PRECIP_AGREEMENT', AGREEMENT_DIRECTORY / 'gridded_dataset_agreement.nc')).expanduser().resolve()
