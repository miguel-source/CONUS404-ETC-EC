"""We assemble native daily PERSIANN-CDR files into expanded event files.

We extracted identify_precip_variable and create_persiann_event_file from our
original PERSIANN-CDR-EVENTS.ipynb (cells 8 and 9). We concatenate daily fields
with exact coordinate matching and assign dates from the NOAA filenames.
We perform no spatial interpolation or temporal aggregation at this stage.
The release CLI rejects ambiguous duplicate daily files explicitly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from study_paths import PROJECT_ROOT
from download_persiann_cdr_events import EVENTS, choose_events

def identify_precip_variable(ds):

    # First check common PERSIANN names
    preferred_names = [
        "precipitation",
        "precip",
        "prec",
        "rain",
        "RAIN",
    ]

    for name in preferred_names:
        if name in ds.data_vars:
            return name

    # Otherwise find a spatial numeric variable
    candidates = []

    for name, da in ds.data_vars.items():

        if np.issubdtype(da.dtype, np.number):

            # Ignore obvious coordinate/bounds variables
            if name.lower() not in {
                "time_bnds",
                "lat_bnds",
                "lon_bnds",
                "crs",
            }:

                candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify precipitation variable. "
        f"Candidate variables: {candidates}"
    )

def create_persiann_event_file(
    event,
    file_index,
    output_dir,
    overwrite=False,
):

    name = event["name"]
    storm_id = event["storm_id"]

    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])

    expected_dates = pd.date_range(
        start=start,
        end=end,
        freq="D"
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Event window: "
        f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    )

    print(
        f"Expected days: {len(expected_dates)}"
    )

    # --------------------------------------------------------
    # Find every daily file
    # --------------------------------------------------------

    missing_dates = [
        date
        for date in expected_dates
        if date not in file_index
    ]

    if missing_dates:

        print("\nMISSING DAYS:")

        for date in missing_dates:
            print("   ", date.strftime("%Y-%m-%d"))

        raise FileNotFoundError(
            f"{name}: {len(missing_dates)} required "
            "PERSIANN daily file(s) are missing."
        )

    selected_files = [
        file_index[date]
        for date in expected_dates
    ]

    print("All daily files found.")

    # --------------------------------------------------------
    # Determine precipitation variable
    # --------------------------------------------------------

    with xr.open_dataset(selected_files[0], engine="netcdf4") as test:

        precip_var = identify_precip_variable(test)

        print("Precipitation variable:", precip_var)
        print("Dimensions:", test[precip_var].dims)
        print(
            "Units:",
            test[precip_var].attrs.get(
                "units",
                "not specified"
            )
        )

    # --------------------------------------------------------
    # Read files one at a time and assign proper date
    # --------------------------------------------------------

    daily_arrays = []

    for date, path in zip(
        expected_dates,
        selected_files
    ):

        ds = xr.open_dataset(
            path,
            engine="netcdf4",
            chunks={}
        )

        da = ds[precip_var]

        # ----------------------------------------------------
        # Remove existing singleton time if necessary
        # ----------------------------------------------------

        if "time" in da.dims:

            if da.sizes["time"] != 1:

                raise RuntimeError(
                    f"{path.name}: expected one daily "
                    f"time step but found {da.sizes['time']}"
                )

            da = da.isel(
                time=0,
                drop=True
            )

        # ----------------------------------------------------
        # Add correct daily timestamp from filename
        # ----------------------------------------------------

        da = da.expand_dims(
            time=[date.to_datetime64()]
        )

        daily_arrays.append(da)

    # --------------------------------------------------------
    # Combine all dates
    # --------------------------------------------------------

    event_da = xr.concat(
        daily_arrays,
        dim="time",
        coords="minimal",
        compat="override",
        join="exact",
    )

    event_da.name = precip_var

    # --------------------------------------------------------
    # Verify time
    # --------------------------------------------------------

    actual_dates = pd.DatetimeIndex(
        event_da.time.values
    )

    if not actual_dates.equals(expected_dates):

        raise RuntimeError(
            f"{name}: final dates do not match "
            "expected event dates."
        )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    event_ds = event_da.to_dataset()

    event_ds.attrs.update({
        "title":
            f"PERSIANN-CDR daily precipitation for Hurricane {name}",

        "event_name":
            name,

        "noaa_storm_id":
            storm_id,

        "event_window_type":
            "expanded",

        "event_window_start_utc":
            f"{start:%Y-%m-%d}T00:00:00Z",

        "event_window_end_utc_inclusive":
            f"{end:%Y-%m-%d}T00:00:00Z",

        "number_of_days":
            len(expected_dates),

        "source_product":
            "PERSIANN-CDR",

        "temporal_resolution":
            "daily",

        "processing":
            "Daily source NetCDF files concatenated by event; "
            "no spatial interpolation or temporal aggregation performed.",
    })

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    output_name = (
        f"PERSIANN-CDR_{name}_{storm_id}_expanded_"
        f"{start:%Y%m%d}_{end:%Y%m%d}_UTC00.nc"
    )

    output_path = (
        output_dir /
        output_name
    )

    if output_path.exists() and not overwrite:

        print(
            "File already exists — skipping:"
        )
        print(output_path)

        # Close source datasets
        for da in daily_arrays:
            try:
                da.close()
            except Exception:
                pass

        return output_path

    # --------------------------------------------------------
    # Compression
    # --------------------------------------------------------

    spatial_dims = [
        d for d in event_da.dims
        if d != "time"
    ]

    chunksizes = [1]

    for dim in spatial_dims:
        chunksizes.append(
            min(
                500,
                event_da.sizes[dim]
            )
        )

    encoding = {
        precip_var: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": tuple(chunksizes),
        }
    }

    # --------------------------------------------------------
    # Write NetCDF
    # --------------------------------------------------------

    print("\nWriting:")
    print(output_path)

    with ProgressBar():

        event_ds.to_netcdf(
            output_path,
            engine="netcdf4",
            encoding=encoding,
        )

    print("Done.")

    # --------------------------------------------------------
    # Validate resulting file
    # --------------------------------------------------------

    with xr.open_dataset(
        output_path, engine="netcdf4"
    ) as check:

        check_dates = pd.DatetimeIndex(
            check.time.values
        )

        if not check_dates.equals(
            expected_dates
        ):
            raise RuntimeError(
                f"{name}: output time validation failed."
            )

        print(
            f"Validated: {len(check_dates)} days"
        )

        print(
            f"{check_dates[0]:%Y-%m-%d} "
            f"to "
            f"{check_dates[-1]:%Y-%m-%d}"
        )

    # --------------------------------------------------------
    # Close source files
    # --------------------------------------------------------

    for da in daily_arrays:
        try:
            da.close()
        except Exception:
            pass

    return output_path

DATE_PATTERN=re.compile(r"PERSIANN-CDR_v\d+r\d+_(\d{8})[-_]c\d{8}\.nc$",re.IGNORECASE)

def index_daily_files(directory):
    file_index={}
    for path in sorted(Path(directory).rglob('*.nc')):
        match=DATE_PATTERN.search(path.name)
        if not match:
            continue
        when=pd.to_datetime(match.group(1),format='%Y%m%d')
        if when in file_index:
            raise ValueError(f'Ambiguous duplicate daily files for {when:%Y-%m-%d}: {file_index[when].name}, {path.name}. Keep the recorded source version explicitly.')
        file_index[when]=path
    if not file_index:
        raise FileNotFoundError(f'No native daily PERSIANN-CDR files found in {directory}')
    return file_index

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily-dir',type=Path,default=PROJECT_ROOT/'Hurricane_Events_P_Data/PERSIANN-CDR/daily')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'Hurricane_Events_P_Data/PERSIANN-CDR/Event_Files')
    parser.add_argument('--events',nargs='*',default=None)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index=index_daily_files(args.daily_dir)
    outputs=[]
    for selected in choose_events(args.events):
        event={'name':selected.name,'storm_id':selected.storm_id,'start':selected.expanded_start.isoformat(),'end':selected.expanded_end.isoformat()}
        result=create_persiann_event_file(event,index,args.output_dir,overwrite=args.overwrite)
        source_names=[index[d].relative_to(args.daily_dir).as_posix() for d in pd.date_range(event['start'],event['end'])]
        outputs.append({'event':event,'output_name':result.name,'native_daily_source_files':source_names,'output_sha256':hashlib.sha256(result.read_bytes()).hexdigest()})
    (args.output_dir/'persiann_event_assembly_manifest.json').write_text(json.dumps({'processing':'We concatenate native daily precipitation without regridding or temporal aggregation.','events':outputs},indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
