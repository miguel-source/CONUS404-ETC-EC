"""We preserve native precipitation values, coordinates, and complete daily dates."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from assemble_persiann_events import create_persiann_event_file,index_daily_files

def write_days(directory,shift_second_grid=False):
    directory.mkdir()
    dates=pd.date_range('2020-01-01',periods=2)
    expected=[]
    for i,when in enumerate(dates):
        values=(np.arange(4,dtype=np.float32).reshape(1,2,2)+i*10)
        if i==0:
            values[0,1,1]=np.nan
        expected.append(values[0].copy())
        longitude=[260.125,260.375] if not (i and shift_second_grid) else [260.25,260.5]
        ds=xr.Dataset({'precipitation':(('time','lat','lon'),values,{'units':'mm','long_name':'native daily precipitation'})},coords={'time':[when+pd.Timedelta(hours=12)],'lat':[30.375,30.125],'lon':longitude})
        ds.to_netcdf(directory/f'PERSIANN-CDR_v01r01_{when:%Y%m%d}_c20210101.nc',engine='netcdf4')
    return np.stack(expected)

def event():
    return {'name':'Synthetic','storm_id':'TEST2020','start':'2020-01-01','end':'2020-01-02'}

def test_native_daily_concatenation_preserves_values_grid_units_and_dates(tmp_path):
    expected=write_days(tmp_path/'daily')
    output=create_persiann_event_file(event(),index_daily_files(tmp_path/'daily'),tmp_path)
    with xr.open_dataset(output,engine='netcdf4') as actual:
        np.testing.assert_allclose(actual.precipitation.values,expected,equal_nan=True)
        np.testing.assert_array_equal(actual.lat,[30.375,30.125])
        np.testing.assert_array_equal(actual.lon,[260.125,260.375])
        np.testing.assert_array_equal(actual.time,pd.date_range('2020-01-01',periods=2))
        assert actual.precipitation.attrs['units']=='mm'
        assert actual.precipitation.dtype==np.float32

def test_missing_daily_file_prevents_event_assembly(tmp_path):
    write_days(tmp_path/'daily')
    index=index_daily_files(tmp_path/'daily')
    del index[pd.Timestamp('2020-01-02')]
    with pytest.raises(FileNotFoundError,match='missing'):
        create_persiann_event_file(event(),index,tmp_path)

def test_mismatched_native_grids_are_not_silently_interpolated(tmp_path):
    write_days(tmp_path/'daily',shift_second_grid=True)
    with pytest.raises(ValueError):
        create_persiann_event_file(event(),index_daily_files(tmp_path/'daily'),tmp_path)
