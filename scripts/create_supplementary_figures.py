"""We create supplemental figures from saved estimates, without rerunning inference.

All inputs are explicit paths. CSV tables preserve plotted values and masks.
Only the CONUS404--CONUS404BA random-error covariance is estimated by EC;
the other error-pair covariances are assumed zero rather than measured zero.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from netCDF4 import Dataset as NetCDFDataset, num2date
import numpy as np
from PIL import Image
import shapely
from brown_green_cartography import PALETTE_ANCHORS, add_cartography

WINDOWS = ("core", "expanded")
LABELS = {"conus404": "CONUS404", "conus404ba": "CONUS404BA", "mrms": "MRMS", "persiann_cdr": "PERSIANN-CDR"}
TITLES = {"S1":"EC correlation between CONUS404 and CONUS404BA random errors", "S2":"Spatial subsets used in collocation comparisons", "S3":"Paired EC bootstrap diagnostics", "S4":"Sensitivity of estimated latent R² to log1p transformation", "S5":"Storm-window sampling", "S6":"Sign of separate-triplet error-variance estimates"}
EXTENT = (-105.5, -71.5, 26., 45.)
GRAY, BROWN, GREEN = "#B9BEC3", "#A6611A", "#238B78"
CMAP = LinearSegmentedColormap.from_list("brown_cream_green", PALETTE_ANCHORS, 1024)
CMAP.set_bad((1, 1, 1, 0))

class ArrayVariable:
    """Expose saved figure-input arrays with their original dimension metadata."""
    def __init__(self,array,metadata):
        self.array=array; self.dimensions=tuple(metadata["dimensions"])
        for name,value in metadata.get("attributes",{}).items(): setattr(self,name,value)
    def __getitem__(self,key): return self.array[key]

class ArrayDataset:
    """Read a compact, pickle-free NPZ figure-input archive."""
    def __init__(self,path):
        self.archive=np.load(path,allow_pickle=False)
        self.metadata=json.loads(str(self.archive["__metadata__"]))
    def __getitem__(self,name): return ArrayVariable(self.archive[name],self.metadata[name])
    def __enter__(self): return self
    def __exit__(self,*exc): self.archive.close()

def Dataset(path):
    return ArrayDataset(path) if Path(path).suffix.lower()==".npz" else NetCDFDataset(path)

def export_bundle(args,destination):
    """We bundle only the saved values needed to reproduce these figures."""
    destination.mkdir(parents=True,exist_ok=True)
    selected={
        "ec":["lat","lon","window","transform","product","error_correlation_model_pair","error_covariance_model_pair","error_variance","valid_primary_ec","bootstrap_valid_fraction","sample_eligible","error_sd_difference_ba_minus_conus404_ci_low","error_sd_difference_ba_minus_conus404_ci_high"],
        "agreement":["lat","lon","window","transform","dataset_pair","study_domain_mask","study_watershed_fraction","valid_agreement"],
        "daily":["lat","lon","date","event_name","storm_id","storm_index","is_core_day"],
    }
    triplet=["lat","lon","window","product","error_variance_unmasked","estimated_r2_to_truth_unmasked","valid_both_triplets","sample_eligible"]
    for name in ("triplet_a_raw","triplet_b_raw","triplet_a_log","triplet_b_log"): selected[name]=triplet
    sources,files={},{}
    for key,names in selected.items():
        path=getattr(args,key); arrays={}; metadata={}
        with NetCDFDataset(path) as ds:
            for name in names:
                var=ds[name]; value=var[:]
                if np.asarray(value).dtype.kind in "OUS": value=np.asarray(value,dtype=str)
                else: value=np.ma.asarray(value,dtype=np.float64).filled(np.nan)
                arrays[name]=value
                metadata[name]={"dimensions":list(var.dimensions),"attributes":{k:var.getncattr(k) for k in ("units","calendar") if k in var.ncattrs()}}
        arrays["__metadata__"]=np.asarray(json.dumps(metadata),dtype=str)
        filename=key+".npz"; np.savez_compressed(destination/filename,**arrays)
        files[key]=filename
        sources[key]={"original_filename":path.name,"original_sha256":sha(path),"bundle_sha256":sha(destination/filename)}
    shutil.copy2(args.boundaries,destination/"map_boundaries.gpkg"); files["boundaries"]="map_boundaries.gpkg"
    sources["boundaries"]={"original_filename":args.boundaries.name,"original_sha256":sha(args.boundaries),"bundle_sha256":sha(destination/"map_boundaries.gpkg")}
    description={"title":"Exact saved inputs for supplemental figure reproduction","format":"NumPy NPZ arrays with JSON dimensions and only units/calendar attributes; allow_pickle=False","inputs":files,"source_provenance":sources,"contains_raw_daily_precipitation":False,"scope":"Derived figure inputs only; reproduce the plots, not the original data extraction or 5000-draw inference"}
    (destination/"figure_inputs_manifest.json").write_text(json.dumps(description,indent=2),encoding="utf-8")
    return description

def numeric(ds, name, **sel):
    var = ds[name]
    key = tuple(list(ds[d][:]).index(sel[d]) if d in sel else slice(None) for d in var.dimensions)
    return np.ma.asarray(var[key], dtype=np.float64).filled(np.nan)

def edges(v):
    return np.r_[v[0]-(v[1]-v[0])/2, (v[1:]+v[:-1])/2, v[-1]+(v[-1]-v[-2])/2]

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def geometries(path, table):
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as con:
        column, epsg = con.execute("SELECT column_name,srs_id FROM gpkg_geometry_columns WHERE table_name=?", (table,)).fetchone()
        if epsg != 4326: raise ValueError("Boundary layers must use EPSG:4326")
        rows = con.execute(f'SELECT "{column}" FROM "{table}"').fetchall()
    result = []
    for (raw,) in rows:
        raw = bytes(raw)
        offset = 8+{0:0,1:32,2:48,3:48,4:64}[(raw[3] >> 1) & 7]
        geom = shapely.from_wkb(raw[offset:])
        if not geom.is_valid: raise ValueError(f"Invalid geometry in {table}")
        result.append(geom)
    return result

def line_parts(geom):
    if geom.geom_type in ("LineString", "LinearRing"): return [np.asarray(geom.coords)]
    return [p for g in geom.geoms for p in line_parts(g)]

def basemap(ax, layers, left=True):
    ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:])
    for name,col,lw in (("states","#999999",.30),("conus","#333333",.65),("watersheds","#111111",.55)):
        ax.add_collection(LineCollection([p for g in layers[name] for p in line_parts(g.boundary)], colors=col, linewidths=lw, zorder=5))
    ax.set_aspect("equal", adjustable="box"); ax.grid(False)
    ax.xaxis.set_major_locator(FixedLocator([-100,-90,-80])); ax.yaxis.set_major_locator(FixedLocator([30,35,40]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{abs(v):.0f}°W")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0f}°N"))
    ax.tick_params(labelsize=11,length=3,width=.6,labelleft=left)
    for spine in ax.spines.values(): spine.set_color("#8B959B"); spine.set_linewidth(.65)
    add_cartography(ax)

def mesh(ax,ctx,values,mask,norm,cmap=CMAP,gray=True):
    if gray:
        badmap=ListedColormap([GRAY]); badmap.set_bad((1,1,1,0))
        ax.pcolormesh(edges(ctx["lon"]),edges(ctx["lat"]),np.where(ctx["domain"] & ~mask,1.,np.nan),cmap=badmap,vmin=0,vmax=1,shading="flat",rasterized=True,edgecolors="none",antialiased=False,zorder=1)
    return ax.pcolormesh(edges(ctx["lon"]),edges(ctx["lat"]),np.where(mask,values,np.nan),cmap=cmap,norm=norm,shading="flat",edgecolors="none",antialiased=False,rasterized=True,zorder=2)

def weighted(ctx,values,mask):
    mask=mask & np.isfinite(values)
    return float(np.sum(values[mask]*ctx["weights"][mask])/ctx["weights"][mask].sum())

def coverage(ctx,mask): return float(100*ctx["weights"][mask].sum()/ctx["weights"].sum())

def table(path,rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def pixel_rows(ctx,window,columns):
    rows=[]
    for y,x in zip(*np.where(ctx["domain"])):
        row={"window":window,"latitude":float(ctx["lat"][y]),"longitude":float(ctx["lon"][x]),"watershed_fraction_coslat_weight":float(ctx["weights"][y,x])}
        for name,arr in columns.items():
            value=arr[y,x]
            row[name]=int(value) if np.issubdtype(arr.dtype,np.bool_) else float(value) if np.isfinite(value) else ""
        rows.append(row)
    return rows

def save_figure(fig,out,stem,caption,data_rows,stats):
    dest=out/stem; dest.mkdir(parents=True,exist_ok=True); fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); clipped=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            if min(box.x0,box.y0)<-.012 or max(box.x1,box.y1)>1.012: clipped.append(text.get_text())
    if clipped: raise ValueError(f"Text outside figure canvas in {stem}: {clipped}")
    files={}
    for ext in ("png","tif","pdf","svg"):
        path=dest/f"{stem}_600dpi.{ext}"
        kwargs={"pil_kwargs":{"compression":"tiff_lzw"}} if ext=="tif" else {}
        fig.savefig(path,dpi=600,**kwargs); files[ext]=path.relative_to(out).as_posix()
    fig.savefig(dest/"preview.png",dpi=120); inches=list(fig.get_size_inches()); plt.close(fig)
    for ext in ("png","tif"):
        with Image.open(out/files[ext]) as raster:
            assert all(abs(v-600)<.1 for v in raster.info["dpi"])
            assert raster.size==tuple(round(v*600) for v in inches)
    table(dest/"figure_data.csv",data_rows); (dest/"caption.md").write_text(caption+"\n",encoding="utf-8")
    meta={"figure":stem,"id":stem.split("_")[0],"title":TITLES[stem.split("_")[0]],"caption":caption,"caption_md":f"{stem}/caption.md","metadata":f"{stem}/metadata.json","files":files,"preview":f"{stem}/preview.png","dpi":600,"font":plt.rcParams["font.family"][0],"figure_inches":inches,"palette":list(PALETTE_ANCHORS),"statistics":stats,"data":f"{stem}/figure_data.csv","source_estimates_modified":False}
    meta.update(files)
    (dest/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta

def two_maps(title,subtitle):
    fig=plt.figure(figsize=(16,6.9),facecolor="white")
    fig.text(.48,.965,title,ha="center",va="top",fontsize=21,weight="bold"); fig.text(.48,.89,subtitle,ha="center",fontsize=13)
    axes=[fig.add_axes([x,.25,.395,.512]) for x in (.055,.505)]
    fig.text(.48,.16,"Longitude",ha="center",fontsize=14); fig.text(.012,.51,"Latitude",ha="center",va="center",rotation=90,fontsize=14)
    return fig,axes

def ec_correlation(args,ctx):
    fig,axes=two_maps("Four-product EC: model-pair random-error correlation","CONUS404 and CONUS404BA  |  Raw daily precipitation")
    rows,stats=[],[]
    with Dataset(args.ec) as ds:
        for i,window in enumerate(WINDOWS):
            rho=numeric(ds,"error_correlation_model_pair",window=window,transform="raw"); mask=ctx["primary"][window]
            assert np.all((rho[mask]>=0)&(rho[mask]<=1))
            covariance=numeric(ds,"error_covariance_model_pair",window=window,transform="raw")
            variance_c=numeric(ds,"error_variance",window=window,transform="raw",product="conus404")
            variance_b=numeric(ds,"error_variance",window=window,transform="raw",product="conus404ba")
            np.testing.assert_allclose(rho[mask],covariance[mask]/np.sqrt(variance_c[mask]*variance_b[mask]),rtol=2e-6,atol=2e-7)
            m=mesh(axes[i],ctx,rho,mask,Normalize(0,1)); basemap(axes[i],ctx["layers"],i==0)
            mean=weighted(ctx,rho,mask)
            axes[i].set_title(f"({'ab'[i]}) {window.capitalize()}",loc="left",fontsize=17,weight="bold",pad=12)
            axes[i].set_title(f"Mean = {mean:.4f}",loc="right",fontsize=13,pad=12)
            axes[i].text(.025,.97,f"{mask.sum()} cells  |  {coverage(ctx,mask):.2f}% watershed area",transform=axes[i].transAxes,fontsize=10,va="top",zorder=30,bbox=dict(facecolor="white",edgecolor="none",alpha=.97,pad=2))
            rows+=pixel_rows(ctx,window,{"error_correlation":rho,"paired_ec_retained":mask})
            stats.append({"window":window,"cells":int(mask.sum()),"coverage_percent":coverage(ctx,mask),"weighted_mean":mean,"minimum":float(rho[mask].min()),"maximum":float(rho[mask].max())})
    cb=fig.colorbar(m,cax=fig.add_axes([.925,.27,.015,.47]),ticks=np.linspace(0,1,6)); cb.set_label(r"Random-error correlation, $\rho_{\varepsilon_C,\varepsilon_B}$",labelpad=12)
    fig.legend(handles=[Patch(facecolor=GRAY,label="Excluded from retained paired EC subset")],loc="lower center",bbox_to_anchor=(.48,.085),frameon=False,fontsize=12)
    fig.text(.48,.035,"Admissible EC estimates and ≥90% admissible bootstrap draws; all displayed correlations are positive.",ha="center",fontsize=11)
    caption="Figure S1. We estimated the correlation between CONUS404 and CONUS404BA random errors using four-product extended collocation (EC), with MRMS Radar-Only QPE and PERSIANN-CDR as the other inputs, for (a) core and (b) expanded event windows. The model permits nonzero CONUS404–CONUS404BA error covariance and assumes the other error pairs are uncorrelated. Colors show random-error correlation, not precipitation correlation or latent-signal R². We retained cells with admissible primary EC estimates and at least 90% admissible storm-block bootstrap draws. Gray denotes excluded study cells. A common linear 0–1 scale contains all displayed estimates; correlations can generally range from −1 to 1. Means use watershed-overlap fraction times cosine-latitude weights. The common grid is 0.25° in EPSG:4326. Watershed boundaries are black and states are gray. WGS84 ruler scale bars are calibrated at 27.35°N; north symbols follow the main figures."
    return save_figure(fig,args.output,"S1_ec_error_correlation",caption,rows,stats)

def selection(args,ctx):
    fig,axes=two_maps("Spatial subsets used in the collocation comparisons","Raw daily precipitation  |  683 center-selected study cells")
    colors=[GRAY,"#DFC27D","#80CDC1","#01665E"]; cmap=ListedColormap(colors); cmap.set_bad((1,1,1,0)); rows,stats=[],[]
    for i,window in enumerate(WINDOWS):
        paired,common=ctx["primary"][window],ctx["common"][window]; code=paired.astype(int)+2*common.astype(int)
        mesh(axes[i],ctx,code,ctx["domain"],Normalize(-.5,3.5),cmap,False); basemap(axes[i],ctx["layers"],i==0)
        axes[i].set_title(f"({'ab'[i]}) {window.capitalize()}",loc="left",fontsize=17,weight="bold",pad=12)
        axes[i].text(.025,.97,f"Paired EC: {paired.sum()} cells; common triplets: {common.sum()}",transform=axes[i].transAxes,fontsize=10.5,va="top",zorder=30,bbox=dict(facecolor="white",edgecolor="none",alpha=.97,pad=2))
        rows+=pixel_rows(ctx,window,{"paired_ec_retained":paired,"common_triplet_admissible":common,"selection_category":code})
        stats.append({"window":window,"paired_ec_cells":int(paired.sum()),"common_triplet_cells":int(common.sum()),"paired_ec_coverage_percent":coverage(ctx,paired),"common_triplet_coverage_percent":coverage(ctx,common),"category_counts":{str(k):int(np.sum(ctx["domain"]&(code==k))) for k in range(4)}})
    names=["Neither subset","Paired EC only","Common triplets only","Both subsets"]
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,names)],loc="lower center",bbox_to_anchor=(.48,.085),ncol=4,frameon=False,fontsize=12)
    fig.text(.48,.035,"The paired EC and common-triplet criteria answer different comparison questions.",ha="center",fontsize=11)
    caption="Figure S2. We compared the spatial subsets used for the primary paired EC analysis and the common-cell triplet summaries during (a) core and (b) expanded windows. Paired EC cells satisfy primary model-block admissibility and at least 90% admissible bootstrap draws; valid reference-product diagnostic variances are not required. Common-triplet cells satisfy full admissibility in both CONUS404–MRMS–PERSIANN-CDR and CONUS404BA–MRMS–PERSIANN-CDR triplets, without bootstrap screening. Colors distinguish membership in the two subsets. These are separate selection criteria, rather than consecutive stages of a single screening procedure. White is outside the center-selected study domain. Boundaries, grid, compass, and scale bars follow Figure S1. The exported data provide both selection masks and category codes."
    return save_figure(fig,args.output,"S2_selection_subsets",caption,rows,stats)

def bootstrap(args,ctx):
    fig=plt.figure(figsize=(16,10.6),facecolor="white")
    fig.text(.48,.97,"Paired EC bootstrap diagnostics",ha="center",va="top",fontsize=22,weight="bold")
    fig.text(.48,.924,"Raw precipitation  |  CONUS404BA minus CONUS404 random-error SD",ha="center",fontsize=13)
    rows,stats,widths=[],[],[]
    with Dataset(args.ec) as ds:
        for window in WINDOWS:
            widths.append(numeric(ds,"error_sd_difference_ba_minus_conus404_ci_high",window=window,transform="raw")-numeric(ds,"error_sd_difference_ba_minus_conus404_ci_low",window=window,transform="raw"))
        widthmax=float(np.ceil(max(np.nanmax(widths[i][ctx["primary"][w]]) for i,w in enumerate(WINDOWS))/5)*5)
        for i,window in enumerate(WINDOWS):
            eligible=numeric(ds,"sample_eligible",window=window)==1
            fraction=numeric(ds,"bootstrap_valid_fraction",window=window,transform="raw")
            width=widths[i]; retained=ctx["primary"][window]
            specs=((fraction,ctx["domain"]&eligible&np.isfinite(fraction),Normalize(0,1),"Admissible-draw fraction"),(width,retained&np.isfinite(width),Normalize(0,widthmax),"Conditional interval width"))
            for j,(values,mask,norm,label) in enumerate(specs):
                ax=fig.add_axes([(.055,.505)[i],(.55,.16)[j],.395,.333])
                m=mesh(ax,ctx,values,mask,norm); basemap(ax,ctx["layers"],i==0)
                ax.set_title(f"({'abcd'[j*2+i]}) {window.capitalize()}: {label}",loc="left",fontsize=14,weight="bold",pad=10)
                ax.text(.025,.97,f"Mean = {weighted(ctx,values,mask):.3f}"+("" if j==0 else " mm day⁻¹"),transform=ax.transAxes,fontsize=10.5,va="top",zorder=30,bbox=dict(facecolor="white",edgecolor="none",alpha=.97,pad=2))
                if i==1:
                    cb=fig.colorbar(m,cax=fig.add_axes([.925,(.57,.18)[j],.015,.285]))
                    cb.set_label("Admissible fraction" if j==0 else "Interval width (mm day⁻¹)",labelpad=10)
            rows+=pixel_rows(ctx,window,{"bootstrap_admissible_fraction":fraction,"sample_eligible":eligible,"paired_ec_retained":retained,"conditional_95_percent_interval_width_mm_day":width})
            stats.append({"window":window,"admissible_fraction_mean_eligible":weighted(ctx,fraction,ctx["domain"]&eligible&np.isfinite(fraction)),"interval_width_mean_retained_mm_day":weighted(ctx,width,retained),"interval_width_max_retained_mm_day":float(np.nanmax(width[retained]))})
    fig.text(.48,.092,"Longitude",ha="center",fontsize=14)
    fig.text(.012,.51,"Latitude",ha="center",va="center",rotation=90,fontsize=14)
    fig.text(.48,.046,"Top: sample-eligible cells with saved bootstrap diagnostics. Bottom: retained paired EC cells.",ha="center",fontsize=11)
    fig.text(.48,.015,"Interval widths are conditional on admissible bootstrap estimates; gray denotes cells outside each displayed subset.",ha="center",fontsize=10.5)
    caption="Figure S3. We examined the fraction of admissible paired storm-block bootstrap draws (a,b) and the width of the conditional 95% bootstrap interval for the CONUS404BA–CONUS404 random-error SD difference (c,d). Left and right columns show core and expanded windows. Admissible fractions are displayed for sample-eligible study cells with saved bootstrap diagnostics. Interval widths, calculated as the 97.5th minus the 2.5th percentile, are displayed only on the retained paired EC subset. Both windows use common scales for the same quantity, without clipping retained values. We used the saved 5,000-draw estimates; this figure does not rerun resampling. The intervals are conditional on admissible EC solutions and are neither simultaneous spatial confidence bands nor a domain-wide significance test. Panel means are watershed-area weighted as in Figure S1. Gray denotes cells outside each displayed subset."
    return save_figure(fig,args.output,"S3_bootstrap_diagnostics",caption,rows,stats)

def log_sensitivity(args,ctx):
    fig,axs=plt.subplots(2,2,figsize=(14,11.5),facecolor="white",sharex=True,sharey=True)
    fig.subplots_adjust(left=.095,right=.96,bottom=.14,top=.875,hspace=.32,wspace=.22)
    fig.suptitle("Sensitivity of estimated latent R² to log1p transformation",y=.97,fontsize=21,weight="bold")
    fig.text(.5,.922,"Identical cells valid in both triplets under both transformations",ha="center",fontsize=13)
    rows,stats=[],[]
    for j,(product,rawpath,logpath,color) in enumerate((("conus404",args.triplet_a_raw,args.triplet_a_log,BROWN),("conus404ba",args.triplet_b_raw,args.triplet_b_log,GREEN))):
        with Dataset(rawpath) as raw,Dataset(logpath) as log:
            for i,window in enumerate(WINDOWS):
                x=numeric(raw,"estimated_r2_to_truth_unmasked",window=window,product=product)
                y=numeric(log,"estimated_r2_to_truth_unmasked",window=window,product=product)
                commonlog=numeric(log,"valid_both_triplets",window=window)==1
                mask=ctx["common"][window]&commonlog&np.isfinite(x)&np.isfinite(y)
                mask&=(x>=0)&(x<=1)&(y>=0)&(y<=1)
                ax=axs[i,j]; ax.plot([0,1],[0,1],color="#555555",ls="--",lw=.8,zorder=1)
                ax.scatter(x[mask],y[mask],s=13,color=color,alpha=.63,linewidths=0,rasterized=True,zorder=2)
                mx,my=weighted(ctx,x,mask),weighted(ctx,y,mask)
                ax.scatter([mx],[my],marker="D",s=64,color=color,edgecolor="black",linewidth=.7,zorder=3)
                ax.set(xlim=(0,1),ylim=(0,1),aspect="equal")
                ax.set_title(f"({'abcd'[i*2+j]}) {window.capitalize()}: {LABELS[product]}",loc="left",fontsize=15,weight="bold",pad=12)
                ax.text(.035,.955,f"n = {mask.sum()} cells\nWeighted means: raw {mx:.3f}; log1p {my:.3f}",transform=ax.transAxes,va="top",fontsize=11)
                ax.grid(False); ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
                for spine in ax.spines.values(): spine.set_linewidth(.7); spine.set_color("#777777")
                rows.extend({"window":window,"product":product,"latitude":float(ctx["lat"][a]),"longitude":float(ctx["lon"][b]),"weight":float(ctx["weights"][a,b]),"raw_estimated_latent_r2":float(x[a,b]),"log1p_estimated_latent_r2":float(y[a,b])} for a,b in zip(*np.where(mask)))
                stats.append({"window":window,"product":product,"common_cells":int(mask.sum()),"weighted_raw_r2":mx,"weighted_log1p_r2":my})
    fig.text(.52,.085,"Estimated latent R²: raw precipitation",ha="center",fontsize=14)
    fig.text(.02,.52,"Estimated latent R²: log1p precipitation",ha="center",va="center",rotation=90,fontsize=14)
    fig.text(.5,.035,"Points: grid cells  |  Diamonds: weighted means  |  Dashed line: equal numerical R²",ha="center",fontsize=12)
    caption="Figure S4. We evaluated transformation sensitivity by comparing estimated latent-signal R² from raw precipitation with estimates from log1p-transformed precipitation for CONUS404 and CONUS404BA in core and expanded windows. We used identical cells satisfying full admissibility in both triplets under both transformations; no bootstrap screening was applied. Each point is a grid cell and diamonds mark watershed-area-weighted means. The dashed line marks equal numerical R². The transformations define different statistical representations of precipitation and associated latent signals, so departures from the diagonal show sensitivity rather than an accuracy improvement. We compare dimensionless R² values, not raw error SD in mm day⁻¹ against transformed-unit error SD."
    return save_figure(fig,args.output,"S4_log1p_sensitivity",caption,rows,stats)

def event_windows(args,ctx):
    from datetime import date
    with Dataset(args.daily) as ds:
        dates=[str(x)[:10] for x in num2date(ds["date"][:],ds["date"].units,calendar=getattr(ds["date"],"calendar","standard"))]
        storms=list(ds["event_name"][:]); ids=list(ds["storm_id"][:])
        blocks=numeric(ds,"storm_index").astype(int); core=numeric(ds,"is_core_day")==1
    dd=[date.fromisoformat(x) for x in dates]; keep={}
    for w in WINDOWS:
        selected=core if w=="core" else np.ones(len(dates),bool); seen=set(); kept=np.zeros(len(dates),bool)
        for i in np.where(selected)[0]:
            if dates[i] not in seen: seen.add(dates[i]); kept[i]=True
        keep[w]=kept
    assert keep["core"].sum()==78 and keep["expanded"].sum()==127
    fig,ax=plt.subplots(figsize=(16,9.2),facecolor="white"); fig.subplots_adjust(left=.17,right=.77,bottom=.16,top=.85)
    fig.suptitle("Storm-window sampling used in the gridded analysis",fontsize=22,weight="bold",y=.97)
    fig.text(.5,.914,"78 unique core dates and 127 unique expanded dates  |  UTC daily accumulations",ha="center",fontsize=14)
    rows,stats,maxday,minday=[],[],0,0
    for s,name in enumerate(storms):
        idx=np.where(blocks==s)[0]; coreidx=idx[core[idx]]; origin=min(dd[i] for i in coreidx)
        x=np.array([(dd[i]-origin).days for i in idx]); maxday,minday=max(maxday,int(x.max())),min(minday,int(x.min()))
        expanded_mask=keep["expanded"][idx]; core_mask=keep["core"][idx]
        ax.scatter(x[expanded_mask],np.full(expanded_mask.sum(),s+.11),marker="s",s=60,color="#DFC27D",edgecolor=BROWN,linewidth=.5,zorder=2)
        ax.scatter(x[core_mask],np.full(core_mask.sum(),s-.11),marker="s",s=60,color=GREEN,edgecolor="#01665E",linewidth=.5,zorder=3)
        dropped=~expanded_mask
        if dropped.any(): ax.scatter(x[dropped],np.full(dropped.sum(),s+.11),marker="o",s=60,facecolor="white",edgecolor="#777777",linewidth=1.1,zorder=4)
        count_core,count_exp=int(core_mask.sum()),int(expanded_mask.sum())
        ax.text(1.025,s,f"{count_core:2d} / {count_exp:2d}",transform=ax.get_yaxis_transform(),va="center",fontsize=12)
        core_start,core_end=min(dd[i] for i in coreidx),max(dd[i] for i in coreidx)
        ax.text(1.19,s,f"{core_start:%d %b}–{core_end:%d %b}",transform=ax.get_yaxis_transform(),va="center",fontsize=11.5)
        for i,rel in zip(idx,x): rows.append({"storm":name,"storm_id":ids[s],"date_utc":dates[i],"days_from_first_core_day":int(rel),"is_archived_core_day":int(core[i]),"retained_core_analysis_date":int(keep["core"][i]),"retained_expanded_analysis_date":int(keep["expanded"][i])})
        stats.append({"storm":name,"storm_id":ids[s],"core_days":count_core,"expanded_days":count_exp,"core_start":str(core_start),"core_end":str(core_end)})
    ax.set_yticks(range(len(storms)),[f"{n} ({str(ids[i])[-4:]})" for i,n in enumerate(storms)])
    ax.set_ylim(len(storms)-.4,-.6); ax.set_xlim(minday-1,maxday+1)
    ax.set_xticks(np.arange(int(np.floor(minday/2)*2),maxday+1,2)); ax.set_xlabel("Days relative to the first core-window day",fontsize=14,labelpad=12)
    ax.tick_params(labelsize=12); ax.grid(False)
    ax.text(1.025,1.04,"Core / exp.",transform=ax.transAxes,fontsize=12,weight="bold")
    ax.text(1.19,1.04,"Core dates",transform=ax.transAxes,fontsize=12,weight="bold")
    for spine in ax.spines.values(): spine.set_color("#777777"); spine.set_linewidth(.7)
    fig.legend(handles=[Patch(facecolor=GREEN,edgecolor="#01665E",label="Core date"),Patch(facecolor="#DFC27D",edgecolor=BROWN,label="Expanded date"),plt.Line2D([],[],marker="o",ls="",markerfacecolor="white",markeredgecolor="#777777",label="Duplicate date omitted from expanded analysis")],loc="lower center",bbox_to_anchor=(.5,.045),ncol=3,frameon=False,fontsize=12)
    caption="Figure S5. We show the UTC calendar-day records contributing to the pooled gridded analysis for each tropical cyclone. Day zero is the first core-window day for the storm; the date ranges at right identify the core intervals. Green and brown squares show core and expanded analysis dates, respectively. Counts at right give retained unique dates for each storm. The archive contains 128 storm–day records, including two records for 4 September 2017. Following the study's all-windows policy, we retained the first EVENTS-order copy and omitted the duplicate expanded record, shown as an open circle, yielding 78 unique core days and 127 unique expanded days. These are pooled storm calendar windows; they do not classify physical storm cores, rainbands, or local impact periods at every pixel. Daily finite, nonnegative matching and cell eligibility were subsequently applied."
    return save_figure(fig,args.output,"S5_storm_window_sampling",caption,rows,stats)

def variance_signs(args,ctx):
    fig,axes=plt.subplots(1,2,figsize=(16,8.3),facecolor="white",sharey=True)
    fig.subplots_adjust(left=.075,right=.97,bottom=.24,top=.83,wspace=.13)
    fig.suptitle("Sign of the separate-triplet random-error variance estimates",fontsize=21,weight="bold",y=.97)
    fig.text(.5,.908,"All 683 sample-eligible study cells  |  Raw daily precipitation",ha="center",fontsize=13)
    rows,stats=[],[]
    for wi,window in enumerate(WINDOWS):
        counts,labels=[],[]
        for ti,path in enumerate((args.triplet_a_raw,args.triplet_b_raw)):
            with Dataset(path) as ds:
                for product in list(ds["product"][:]):
                    values=numeric(ds,"error_variance_unmasked",window=window,product=product)
                    eligible=ctx["domain"]&(numeric(ds,"sample_eligible",window=window)==1)
                    pos=eligible&np.isfinite(values)&(values>=0); neg=eligible&np.isfinite(values)&(values<0); na=eligible&~np.isfinite(values)
                    n=int(eligible.sum()); assert n==683
                    c=[int(pos.sum()),int(neg.sum()),int(na.sum())]; assert sum(c)==n
                    counts.append(c); labels.append(LABELS[product]+f"\nTriplet {'AB'[ti]}")
                    row={"window":window,"triplet":"AB"[ti],"product":product,"sample_eligible_cells":n,"nonnegative_cells":c[0],"negative_cells":c[1],"unavailable_cells":c[2],"negative_percent_of_eligible_cells":100*c[1]/n}
                    rows.append(row); stats.append(row)
        cc=np.asarray(counts); pct=cc/cc.sum(axis=1)[:,None]*100; bottom=np.zeros(6)
        for k,color in enumerate((GREEN,BROWN,GRAY)):
            axes[wi].bar(np.arange(6),pct[:,k],bottom=bottom,width=.7,color=color,edgecolor="white",linewidth=.4)
            for j in range(6):
                if pct[j,k]>=3: axes[wi].text(j,bottom[j]+pct[j,k]/2,str(cc[j,k]),ha="center",va="center",color="white",fontsize=11.5)
            bottom+=pct[:,k]
        axes[wi].set_title(f"({'ab'[wi]}) {window.capitalize()}",loc="left",fontsize=17,weight="bold",pad=13)
        axes[wi].set_xticks(range(6),labels,rotation=28,ha="right",fontsize=10.5)
        axes[wi].set(ylim=(0,100),yticks=np.arange(0,101,20)); axes[wi].grid(False)
        for spine in axes[wi].spines.values(): spine.set_color("#777777"); spine.set_linewidth(.7)
    axes[0].set_ylabel("Share of sample-eligible grid cells (%)",fontsize=14)
    fig.legend(handles=[Patch(facecolor=GREEN,label="Nonnegative variance"),Patch(facecolor=BROWN,label="Negative variance"),Patch(facecolor=GRAY,label="Unavailable estimate")],loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=12)
    fig.text(.5,.025,"Labels give cell counts; a nonnegative variance alone does not establish full-triplet admissibility.",ha="center",fontsize=11.5)
    caption="Figure S6. We summarized the sign of the unmasked raw-precipitation random-error variance estimates for each product in the separate triplets during core and expanded windows. Triplet A contains CONUS404, MRMS Radar-Only QPE, and PERSIANN-CDR; triplet B replaces CONUS404 with CONUS404BA. Bar heights represent unweighted percentages of the 683 sample-eligible study cells; labels give counts. Green denotes finite nonnegative variance, brown denotes finite negative variance, and gray denotes an unavailable algebraic estimate. Negative estimates are nonphysical statistical solutions, not negative actual error variances or zero-error cells. A nonnegative estimate for one product does not establish admissibility of all three products in its triplet. The CSV includes all counts and percentages, including the single unavailable MRMS core estimate in each triplet."
    return save_figure(fig,args.output,"S6_variance_sign_diagnostics",caption,rows,stats)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("ec","agreement","daily","triplet-a-raw","triplet-b-raw","triplet-a-log","triplet-b-log","boundaries"): parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--input-bundle",type=Path,help="Directory containing figure_inputs_manifest.json and compact saved arrays")
    parser.add_argument("--export-input-bundle",type=Path,help="Export exact plotted inputs as compact pickle-free NPZ files")
    parser.add_argument("--bundle-only",action="store_true",help="Export the requested bundle without redrawing figures")
    parser.add_argument("--font",default=os.environ.get("TC_PRECIP_FONT","Times New Roman"),help="Installed font family; defaults to TC_PRECIP_FONT or Times New Roman")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    inputnames=("ec","agreement","daily","triplet_a_raw","triplet_b_raw","triplet_a_log","triplet_b_log","boundaries")
    bundle=None
    if args.input_bundle:
        bundle=json.loads((args.input_bundle/"figure_inputs_manifest.json").read_text(encoding="utf-8"))
        for key in inputnames:
            candidate=(args.input_bundle/bundle["inputs"][key]).resolve()
            if not candidate.is_relative_to(args.input_bundle.resolve()): raise ValueError("Figure-input path escapes its bundle")
            if sha(candidate)!=bundle["source_provenance"][key]["bundle_sha256"]: raise ValueError(f"Figure-input checksum mismatch: {key}")
            setattr(args,key,candidate)
    missing=[k for k in inputnames if getattr(args,k) is None]
    if missing: parser.error("Supply --input-bundle or all explicit inputs; missing: "+", ".join(missing))
    if args.export_input_bundle:
        export_bundle(args,args.export_input_bundle)
        if args.bundle_only:
            print(json.dumps({"exported_figure_input_bundle":str(args.export_input_bundle)}),flush=True); return
    elif args.bundle_only: parser.error("--bundle-only requires --export-input-bundle")
    sources={key:{"filename":getattr(args,key).name,"sha256":sha(getattr(args,key)),"bytes":getattr(args,key).stat().st_size} for key in inputnames}
    plt.rcParams.update({"font.family":args.font,"font.size":12,"mathtext.fontset":"stix","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
    font_manager.findfont(args.font,fallback_to_default=False)
    with Dataset(args.agreement) as ag,Dataset(args.ec) as ec,Dataset(args.triplet_a_raw) as tr:
        ctx={"lat":numeric(ec,"lat"),"lon":numeric(ec,"lon"),"domain":numeric(ag,"study_domain_mask")==1,"primary":{},"common":{}}
        ctx["weights"]=numeric(ag,"study_watershed_fraction")*np.cos(np.deg2rad(ctx["lat"]))[:,None]
        assert ctx["domain"].sum()==683
        np.testing.assert_array_equal(ctx["lat"],numeric(ag,"lat")); np.testing.assert_array_equal(ctx["lon"],numeric(ag,"lon"))
        for key in ("daily","triplet_a_raw","triplet_b_raw","triplet_a_log","triplet_b_log"):
            with Dataset(getattr(args,key)) as comparison:
                np.testing.assert_array_equal(ctx["lat"],numeric(comparison,"lat")); np.testing.assert_array_equal(ctx["lon"],numeric(comparison,"lon"))
        for window in WINDOWS:
            ctx["primary"][window]=(numeric(ec,"valid_primary_ec",window=window,transform="raw")==1)&(numeric(ec,"bootstrap_valid_fraction",window=window,transform="raw")>=.90)
            agreed=numeric(ag,"valid_agreement",window=window,transform="raw",dataset_pair="conus404__conus404ba")==1
            np.testing.assert_array_equal(ctx["primary"][window],agreed)
            ctx["common"][window]=numeric(tr,"valid_both_triplets",window=window)==1
    ctx["layers"]={"states":geometries(args.boundaries.resolve(),"conus_states"),"conus":geometries(args.boundaries.resolve(),"conus_boundary"),"watersheds":geometries(args.boundaries.resolve(),"watersheds_hurricane_analysis")}
    assert len(ctx["layers"]["watersheds"])==23
    results=[]
    for function in (ec_correlation,selection,bootstrap,log_sensitivity,event_windows,variance_signs):
        result=function(args,ctx); results.append(result); print(json.dumps({"completed":result["figure"],"statistics":result["statistics"]}),flush=True)
    assert sources=={key:{"filename":getattr(args,key).name,"sha256":sha(getattr(args,key)),"bytes":getattr(args,key).stat().st_size} for key in inputnames}
    manifest={"title":"Supplemental figures for tropical-cyclone precipitation collocation","figure_count":len(results),"source_provenance":sources,"bundle_provenance":bundle,"sources_modified":False,"collocation_or_bootstrap_rerun":False,"map_crs":"EPSG:4326","map_resolution_degrees":.25,"map_weighting":"study_watershed_fraction * cos(latitude)","figures":results}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (args.output/"captions.md").write_text("\n\n".join(r["caption"] for r in results)+"\n",encoding="utf-8")

if __name__=="__main__": main()
