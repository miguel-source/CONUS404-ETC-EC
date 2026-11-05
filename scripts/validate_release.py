"""We verify release-code hashes and optionally compare original scientific bodies."""
from __future__ import annotations
import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[1]
FUNCTIONS={
    'run_gridded_extended_collocation.py':['_safe_divide','_safe_sqrt','_valid_positive_cross','_metrics_from_covariance','_storm_sufficient_statistics','_covariance_from_statistics','_point_statistics','_observation_indices','_bootstrap_summaries','_loso_metrics'],
    'run_separate_triplet_error_maps.py':['triplet_metrics'],
    'run_gridded_dataset_agreement.py':['_selected_observations','_matched_sample'],
    'prepare_gridded_persiann_etc.py':['build_target_grid','build_sparse_remap','apply_sparse_remap'],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def functions(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,help='Optional original workspace, used only for the original-function audit.')
    parser.add_argument('--refresh-manifest',action='store_true',help='Record hashes after intentional release changes.')
    args=parser.parse_args()
    inventory_path=REPO/'provenance/source_inventory.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
    if args.refresh_manifest:
        for row in inventory:
            row['release_sha256']=sha(REPO/row['destination'])
        inventory_path.write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    mismatches=[r['destination'] for r in inventory if sha(REPO/r['destination'])!=r['release_sha256']]
    if mismatches:
        raise AssertionError(f'Release code differs from recorded hashes: {mismatches}')
    code=[p for folder in ('scripts','src','tests','examples','tools') for p in (REPO/folder).rglob('*.py')]
    leaks=[]
    for path in code:
        text=path.read_text(encoding='utf-8-sig')
        ast.parse(text)
        personal_pattern=r'(?i)(?:[a-z]:[\\/]+Users[\\/]|/' + r'(?:home|Users)/[^/\s]+/)'
        if re.search(personal_pattern,text):
            leaks.append(path.relative_to(REPO).as_posix())
    if leaks:
        raise AssertionError(f'Personal absolute paths remain: {leaks}')
    integrity=[]
    if args.source_root:
        for name,names in FUNCTIONS.items():
            original=functions(args.source_root/'outputs'/name)
            released=functions(REPO/'scripts'/name)
            for function in names:
                assert original[function]==released[function],(name,function)
                integrity.append({'script':name,'function':function,'unchanged':True})
    versions={}
    for name in ['numpy','scipy','pandas','xarray','netCDF4','matplotlib','Pillow','shapely','pyproj','rasterio','geopandas','rioxarray','pytest','intake','intake-xarray','zarr','numcodecs']:
        try:
            versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name]=None
    report={'code_files_parsed':len(code),'recorded_study_sources':len(inventory),'recorded_source_hashes_match':True,'personal_absolute_paths_found':False,'original_function_integrity':integrity,'validation_environment':{'python':sys.version.split()[0],'packages':versions},'scope':'We validated portable code and available synthetic/summary outputs. We did not re-download all source archives, solve a fresh environment on every OS, or rerun the full 5000-replicate observational analysis.'}
    (REPO/'provenance/code_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest=[{'file':p.relative_to(REPO).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(code)]
    (REPO/'provenance/release_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'code_files_parsed':len(code),'scientific_functions_compared':len(integrity),'recorded_sources_checked':len(inventory),'personal_paths_found':False},indent=2))

if __name__=='__main__':
    main()
