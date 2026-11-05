"""We expose single-layer boundary files for unambiguous workflow inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sqlite3

def prepare(source, destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for layer,name in [('watersheds_hurricane_analysis','study_watersheds.gpkg'),('conus_states','conus_states.gpkg')]:
        target=destination/name
        if target.exists():
            raise FileExistsError(f'We do not overwrite existing boundary files: {target}')
        # We copy the original GeoPackage geometries byte for byte. No GDAL
        # driver, coordinate transformation, or spatial-index rebuild is needed.
        with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as original, sqlite3.connect(target) as output:
            output.execute('PRAGMA application_id=1196444487')
            output.execute('PRAGMA user_version=10300')
            for table in ['gpkg_spatial_ref_sys','gpkg_contents','gpkg_geometry_columns',layer]:
                sql=original.execute('SELECT sql FROM sqlite_master WHERE type=\'table\' AND name=?',(table,)).fetchone()
                if not sql:
                    raise ValueError(f'Missing required GeoPackage table: {table}')
                output.execute(sql[0])
                query=f'SELECT * FROM "{table}"'
                parameters=()
                if table in {'gpkg_contents','gpkg_geometry_columns'}:
                    query+=' WHERE table_name=?'
                    parameters=(layer,)
                rows=original.execute(query,parameters).fetchall()
                if not rows:
                    raise ValueError(f'Empty required boundary table: {table}')
                output.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in rows[0])})',rows)
            assert output.execute('SELECT count(*) FROM gpkg_contents').fetchone()[0]==1
        print(target)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.input.resolve(),args.output_dir)

if __name__=='__main__':
    main()
