"""We retain complete HURDAT2 records for the ten storms in our study."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

STORM_IDS=('AL142016','AL092017','AL112017','AL062018','AL112019','AL132020','AL282020','AL292020','AL092021','AL092022')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    lines=args.input.read_text(encoding='utf-8').splitlines(keepends=True)
    records={}
    i=0
    while i<len(lines):
        fields=[p.strip() for p in lines[i].split(',')]
        if len(fields)>=3 and fields[0].startswith('AL') and len(fields[0])==8:
            count=int(fields[2])
            if fields[0] in STORM_IDS:
                records[fields[0]]=lines[i:i+count+1]
            i+=count+1
        else:
            i+=1
    if set(records)!=set(STORM_IDS):
        raise ValueError(f'Missing study storms: {set(STORM_IDS)-set(records)}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(line for key in STORM_IDS for line in records[key]),encoding='utf-8')
    provenance={'description':'We subset complete original HURDAT2 records without changing coordinates, statuses, times, or values.','source_archive_name':args.input.name,'source_archive_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'subset_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'storm_ids':STORM_IDS,'source':'NOAA NHC Atlantic HURDAT2, https://www.nhc.noaa.gov/data/'}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(args.output)

if __name__=='__main__':
    main()
