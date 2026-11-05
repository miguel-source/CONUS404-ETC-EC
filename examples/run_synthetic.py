"""We demonstrate the gridded EC pipeline on simulated, nonnegative rainfall.

This small example is a software check, not evidence about real products.
We disable geographic overlays because the input domain is synthetic.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
import run_gridded_extended_collocation as ec

def make_input(destination, seed=404):
    rng=np.random.default_rng(seed)
    storms,days,ny,nx=10,12,2,2
    n=storms*days
    truth=20+rng.gamma(3,8,(n,ny,nx))
    shared=rng.normal(0,1.5,(n,ny,nx))
    errors=[shared+rng.normal(0,.7,(n,ny,nx)),1.1*shared+rng.normal(0,.8,(n,ny,nx)),rng.normal(0,1.0,(n,ny,nx)),rng.normal(0,1.3,(n,ny,nx))]
    slopes=[1.,1.08,.95,.9]
    products=['conus404','conus404ba','mrms','persiann_cdr']
    variables={p:(('observation','lat','lon'),np.maximum(slopes[i]*truth+errors[i],0)) for i,p in enumerate(products)}
    dates=np.concatenate([pd.date_range(f'{2000+i}-08-01',periods=days).values for i in range(storms)])
    variables.update(date=('observation',dates),storm_index=('observation',np.repeat(np.arange(storms),days)),storm_id=('storm',[f'SYNTHETIC{i:02}' for i in range(storms)]),event_name=('storm',[f'Synthetic {i+1}' for i in range(storms)]),is_core_day=('observation',np.tile(np.arange(days)<8,storms).astype('int8')),study_domain_mask=(('lat','lon'),np.array([[1,1],[1,0]],dtype='int8')),study_watershed_fraction=(('lat','lon'),np.ones((ny,nx))),valid_all4=(('observation','lat','lon'),np.ones((n,ny,nx),dtype='int8')))
    ds=xr.Dataset(variables,coords={'observation':np.arange(n),'storm':np.arange(storms),'lat':[31.125,30.875],'lon':[-96.125,-95.875]},attrs={'description':'Synthetic demonstration; not observations or study results','daily_interval':'[D 00:00 UTC, D+1 00:00 UTC)'})
    destination.parent.mkdir(parents=True,exist_ok=True)
    ds.to_netcdf(destination,engine='netcdf4')
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'results/synthetic')
    parser.add_argument('--bootstrap',type=int,default=64)
    args=parser.parse_args()
    source=make_input(args.output/'synthetic_daily.nc')
    ec.add_map_boundaries=lambda *a,**k:None
    paths=ec.run_analysis(source,args.output/'ec',bootstrap=args.bootstrap,seed=404,do_loso=True)
    result=xr.open_dataset(paths['estimates'],engine='netcdf4') if 'estimates' in paths else xr.open_dataset(args.output/'ec/gridded_extended_collocation_all_windows.nc',engine='netcdf4')
    assert result.sizes['lat']==2 and result.sizes['lon']==2
    assert int(result.sample_eligible.sel(window='core').sum())==3
    assert int(result.sample_eligible.sel(window='expanded').sum())==3
    result.close()
    print(json.dumps({'status':'passed','purpose':'synthetic end-to-end EC example','bootstrap_replicates':args.bootstrap},indent=2))

if __name__=='__main__':
    main()
