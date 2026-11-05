"""We run the archived study stages with explicit paths and recorded commands."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/study.example.json')
    parser.add_argument('--stages', nargs='+', choices=['acquire','assemble','windows','regrid','ec','triplets','paired-support','direct-agreement','summaries','figures'], default=['ec','triplets','paired-support','direct-agreement'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    def path(name):
        value = Path(config[name]).expanduser()
        return value if value.is_absolute() else (REPO / value).resolve()
    project, results, daily = path('project_root'), path('results_root'), path('daily_input')
    ec_dir = results / 'Hurricane_gridded_EC_results'
    ec_file = ec_dir / 'gridded_extended_collocation_all_windows.nc'
    loso_file = ec_dir / 'gridded_extended_collocation_all_windows_leave_one_storm_out.nc'
    agreement_dir = results / 'Hurricane_gridded_agreement_results'
    env = os.environ.copy()
    env.update(TC_PRECIP_PROJECT=str(project), TC_PRECIP_RESULTS=str(results), TC_PRECIP_DAILY=str(daily), TC_PRECIP_EC=str(ec_file), TC_PRECIP_AGREEMENT=str(agreement_dir/'gridded_dataset_agreement.nc'), TC_PRECIP_WATERSHEDS=str(path('watersheds')), TC_PRECIP_STATES=str(path('states')), MPLBACKEND='Agg')
    commands = {
        'acquire': [[name+'.py'] for name in ['download_conus404_events','download_conus404ba_events','download_mrms_events','download_persiann_cdr_events','assemble_persiann_events']],
        'assemble': [['assemble_persiann_events.py']],
        'windows': [['build_tc_event_windows_spherical.py', str(path('watersheds')), str(path('hurdat_input')), str(results/'windows/tc_event_windows_by_watershed_spherical.csv')],['derive_event_table.py','--input',str(results/'windows/tc_event_windows_by_watershed_spherical.csv'),'--output',str(results/'windows/events.csv')]],
        'regrid': [['prepare_gridded_persiann_etc.py','--output-dir',str(daily.parent)]],
        'ec': [['run_gridded_extended_collocation.py','--input',str(daily),'--output-dir',str(ec_dir),'--sample-mode',config['sample_mode'],'--bootstrap',str(config['bootstrap_replicates']),'--seed',str(config['bootstrap_seed']),'--min-storms',str(config['minimum_storms']),'--min-core-observations',str(config['minimum_core_days']),'--min-expanded-observations',str(config['minimum_expanded_days']),'--min-days-per-storm',str(config['minimum_days_per_storm'])]],
        'triplets': [['run_separate_triplet_error_maps.py','--input',str(daily),'--ec-source',str(ec_file),'--output-dir',str(results/'separate_triplet_results')]],
        'paired-support': [['run_gridded_dataset_agreement.py','--daily-input',str(daily),'--ec-input',str(ec_file),'--loso-input',str(loso_file),'--output-dir',str(agreement_dir),'--minimum-bootstrap-valid-fraction',str(config['minimum_bootstrap_valid_fraction'])]],
        'direct-agreement': [['extract_direct_reference_agreement.py']],
        'summaries': [['extract_word_results_metrics.py']],
        'figures': [[name+'.py'] for name in ['create_all_available_triplet_variance_figures','create_all_available_triplet_sd_r2_figures','create_triplet_brown_green_600dpi','create_daily_agreement_violins_600dpi','create_paired_bootstrap_categories_600dpi','create_paired_loso_600dpi']],
    }
    log=[]
    for stage in args.stages:
        for command in commands[stage]:
            full=[sys.executable, str(SCRIPTS/command[0]), *command[1:]]
            print(json.dumps({'stage':stage,'command':full}), flush=True)
            if not args.dry_run:
                subprocess.run(full, cwd=REPO, env=env, check=True)
            log.append({'stage':stage,'command':full,'executed':not args.dry_run})
    if not args.dry_run:
        results.mkdir(parents=True, exist_ok=True)
        (results/'pipeline_run.json').write_text(json.dumps({'configuration':config,'commands':log},indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
