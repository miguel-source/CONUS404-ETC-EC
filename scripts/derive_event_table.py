"""We derive event-wide windows from the watershed-specific contact inventory."""
from __future__ import annotations
import argparse
import csv
from datetime import date,timedelta
from pathlib import Path
from prepare_study_track_subset import STORM_IDS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.input.open(encoding='utf-8-sig',newline='') as handle:
        inventory=list(csv.DictReader(handle))
    rows=[]
    for storm_id in STORM_IDS:
        basins=[r for r in inventory if r['storm_id']==storm_id and r['auto_window_status']=='ok']
        if not basins:
            raise ValueError(f'No valid basin windows for {storm_id}')
        start=min(date.fromisoformat(r['analysis_start_date']) for r in basins)
        end=max(date.fromisoformat(r['analysis_end_date']) for r in basins)
        expanded_start=start-timedelta(days=4)
        expanded_end=end+timedelta(days=1)
        rows.append({'event_name':basins[0]['storm_name'],'storm_id':storm_id,'year':basins[0]['storm_year'],'watershed_count':len(basins),'core_start':start,'core_end':end,'core_days':(end-start).days+1,'expanded_start':expanded_start,'expanded_end':expanded_end,'expanded_days':(expanded_end-expanded_start).days+1})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)

if __name__=='__main__':
    main()
