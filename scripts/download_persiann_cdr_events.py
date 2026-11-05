#!/usr/bin/env python3
"""Download daily NOAA PERSIANN-CDR files for selected hurricane windows.

The default window is the expanded window supplied in the study event table.
Each requested date represents [00:00 UTC, next-day 00:00 UTC). Native daily
grids are preserved unchanged; watershed clipping/regridding belongs in a
separate processing step so the raw download remains reproducible.

The script uses only the Python standard library. It discovers the complete
NOAA filename from each yearly directory because the trailing cYYYYMMDD token
is a file-creation date and cannot safely be guessed.

Examples
--------
Preview all expanded-window downloads without writing files:

    py download_persiann_cdr_events.py --dry-run

Download all expanded windows to the default project directory:

    py download_persiann_cdr_events.py

Download only Ian, or use the narrower core-event dates:

    py download_persiann_cdr_events.py --event Ian
    py download_persiann_cdr_events.py --window core
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import csv
import hashlib
import html
import os
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://www.ncei.noaa.gov/data/precipitation-persiann/access/"
PRODUCT_VERSION = "v01r01"
DEFAULT_DESTINATION = (PROJECT_ROOT / 'Hurricane_Events_P_Data/PERSIANN-CDR')
USER_AGENT = (
    "PERSIANN-CDR-event-downloader/1.0 "
    "(research use; NOAA NCEI public archive)"
)

# NetCDF-3 classic/64-bit variants and NetCDF-4 (HDF5) signatures.
NETCDF_SIGNATURES = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")

# NOAA currently uses an underscore before cYYYYMMDD. Accept a hyphen too so
# the discovery step remains safe if an older/reprocessed filename uses it.
FILE_RE = re.compile(
    rf'href=["\'](?P<filename>PERSIANN-CDR_{PRODUCT_VERSION}_'
    r'(?P<observation>\d{8})[-_]c(?P<creation>\d{8})\.nc)["\']',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Event:
    name: str
    year: int
    storm_id: str
    watershed_count: int
    core_start: date
    core_end: date
    expected_core_days: int
    expanded_start: date
    expanded_end: date
    expected_expanded_days: int

    def bounds(self, window: str) -> tuple[date, date]:
        if window == "core":
            return self.core_start, self.core_end
        if window == "expanded":
            return self.expanded_start, self.expanded_end
        raise ValueError(f"Unknown window: {window}")


EVENTS = (
    Event(
        "Matthew", 2016, "AL142016", 4,
        date(2016, 10, 5), date(2016, 10, 12), 8,
        date(2016, 10, 1), date(2016, 10, 13), 13,
    ),
    Event(
        "Harvey", 2017, "AL092017", 8,
        date(2017, 8, 23), date(2017, 9, 3), 12,
        date(2017, 8, 19), date(2017, 9, 4), 17,
    ),
    Event(
        "Irma", 2017, "AL112017", 3,
        date(2017, 9, 8), date(2017, 9, 14), 7,
        date(2017, 9, 4), date(2017, 9, 15), 12,
    ),
    Event(
        "Florence", 2018, "AL062018", 3,
        date(2018, 9, 12), date(2018, 9, 20), 9,
        date(2018, 9, 8), date(2018, 9, 21), 14,
    ),
    Event(
        "Imelda", 2019, "AL112019", 2,
        date(2019, 9, 16), date(2019, 9, 21), 6,
        date(2019, 9, 12), date(2019, 9, 22), 11,
    ),
    Event(
        "Laura", 2020, "AL132020", 1,
        date(2020, 8, 25), date(2020, 8, 30), 6,
        date(2020, 8, 21), date(2020, 8, 31), 11,
    ),
    Event(
        "Zeta", 2020, "AL282020", 1,
        date(2020, 10, 27), date(2020, 10, 31), 5,
        date(2020, 10, 23), date(2020, 11, 1), 10,
    ),
    Event(
        "Eta", 2020, "AL292020", 1,
        date(2020, 11, 7), date(2020, 11, 15), 9,
        date(2020, 11, 3), date(2020, 11, 16), 14,
    ),
    Event(
        "Ida", 2021, "AL092021", 9,
        date(2021, 8, 27), date(2021, 9, 4), 9,
        date(2021, 8, 23), date(2021, 9, 5), 14,
    ),
    Event(
        "Ian", 2022, "AL092022", 1,
        date(2022, 9, 26), date(2022, 10, 2), 7,
        date(2022, 9, 22), date(2022, 10, 3), 12,
    ),
)


@dataclass(frozen=True)
class RemoteFile:
    observation_date: date
    creation_date: date
    filename: str
    url: str


@dataclass(frozen=True)
class DownloadResult:
    remote: RemoteFile
    local_path: Path
    status: str
    size_bytes: int | None
    sha256: str
    retrieved_utc: str
    error: str = ""


def inclusive_dates(start: date, end: date) -> Iterable[date]:
    """Yield all dates from start through end, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def validate_event_table() -> None:
    """Fail immediately if a transcribed date window disagrees with its count."""
    for event in EVENTS:
        core_days = (event.core_end - event.core_start).days + 1
        expanded_days = (event.expanded_end - event.expanded_start).days + 1
        if core_days != event.expected_core_days:
            raise RuntimeError(
                f"{event.name}: core window has {core_days} days; "
                f"the table says {event.expected_core_days}."
            )
        if expanded_days != event.expected_expanded_days:
            raise RuntimeError(
                f"{event.name}: expanded window has {expanded_days} days; "
                f"the table says {event.expected_expanded_days}."
            )


def retry_delay(attempt: int) -> float:
    """Short exponential backoff capped at 30 seconds."""
    return min(2.0 ** attempt, 30.0)


def should_retry_http(code: int) -> bool:
    return code in {408, 429, 500, 502, 503, 504}


def read_url(url: str, timeout: float, retries: int) -> bytes:
    """Read a small URL, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
            )
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if not should_retry_http(exc.code) or attempt == retries:
                raise
        except (URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == retries:
                raise
        time.sleep(retry_delay(attempt))
    raise RuntimeError(f"Unable to read {url}: {last_error}")


def discover_year(year: int, timeout: float, retries: int) -> dict[date, RemoteFile]:
    """Discover exact native filenames for one year from the NOAA index."""
    year_url = urljoin(BASE_URL, f"{year}/")
    page = html.unescape(read_url(year_url, timeout, retries).decode("utf-8", "replace"))
    candidates: dict[date, list[RemoteFile]] = {}

    for match in FILE_RE.finditer(page):
        observation = datetime.strptime(match.group("observation"), "%Y%m%d").date()
        creation = datetime.strptime(match.group("creation"), "%Y%m%d").date()
        filename = match.group("filename")
        remote = RemoteFile(
            observation_date=observation,
            creation_date=creation,
            filename=filename,
            url=urljoin(year_url, quote(filename)),
        )
        candidates.setdefault(observation, []).append(remote)

    if not candidates:
        raise RuntimeError(f"No {PRODUCT_VERSION} NetCDF files found in {year_url}")

    selected: dict[date, RemoteFile] = {}
    for observation, versions in candidates.items():
        # If NOAA exposes more than one creation for a data date, choose the
        # newest and retain its exact creation date in the manifest.
        selected[observation] = max(
            versions,
            key=lambda item: (item.creation_date, item.filename),
        )
    return selected


def has_netcdf_signature(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("rb") as stream:
            signature = stream.read(8)
        return any(signature.startswith(item) for item in NETCDF_SIGNATURES)
    except OSError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(
    remote: RemoteFile,
    destination: Path,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> DownloadResult:
    """Download one native file through an atomic .part file."""
    target = destination / "daily" / str(remote.observation_date.year) / remote.filename
    retrieved_utc = datetime.now(timezone.utc).isoformat()

    if target.exists() and not overwrite and has_netcdf_signature(target):
        return DownloadResult(
            remote=remote,
            local_path=target,
            status="skipped_valid_existing",
            size_bytes=target.stat().st_size,
            sha256=sha256_file(target),
            retrieved_utc=retrieved_utc,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            if partial.exists():
                partial.unlink()

            request = Request(
                remote.url,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
            )
            digest = hashlib.sha256()
            bytes_written = 0

            with urlopen(request, timeout=timeout) as response, partial.open("wb") as stream:
                content_length_text = response.headers.get("Content-Length")
                expected_bytes = (
                    int(content_length_text) if content_length_text is not None else None
                )
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())

            if expected_bytes is not None and bytes_written != expected_bytes:
                raise OSError(
                    f"Content-Length mismatch: expected {expected_bytes}, "
                    f"received {bytes_written}"
                )
            if not has_netcdf_signature(partial):
                raise OSError("Downloaded response is not a recognizable NetCDF file")

            os.replace(partial, target)
            return DownloadResult(
                remote=remote,
                local_path=target,
                status="downloaded",
                size_bytes=bytes_written,
                sha256=digest.hexdigest(),
                retrieved_utc=datetime.now(timezone.utc).isoformat(),
            )

        except HTTPError as exc:
            last_error = exc
            permanent = not should_retry_http(exc.code)
            if permanent or attempt == retries:
                break
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break

        time.sleep(retry_delay(attempt))

    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass

    return DownloadResult(
        remote=remote,
        local_path=target,
        status="failed",
        size_bytes=None,
        sha256="",
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        error=str(last_error or "Unknown download error"),
    )


def write_manifest(
    destination: Path,
    selected_events: list[Event],
    window: str,
    remote_files: dict[date, RemoteFile],
    results: dict[date, DownloadResult],
) -> Path:
    """Write one row per event-date membership, including shared dates."""
    manifest_dir = destination / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"persiann_cdr_{window}_event_manifest.csv"
    temporary = manifest.with_name(manifest.name + ".tmp")

    fields = [
        "event_name",
        "storm_year",
        "noaa_storm_id",
        "watershed_count",
        "window_type",
        "window_start_utc",
        "window_end_utc_inclusive",
        "core_start_utc",
        "core_end_utc_inclusive",
        "observation_date_utc",
        "interval_start_utc",
        "interval_end_utc_exclusive",
        "is_core_event_date",
        "product",
        "product_version",
        "source_creation_date",
        "source_url",
        "remote_filename",
        "local_path",
        "status",
        "size_bytes",
        "sha256",
        "retrieved_utc",
        "error",
    ]

    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in selected_events:
            start, end = event.bounds(window)
            for observation in inclusive_dates(start, end):
                remote = remote_files[observation]
                result = results[observation]
                next_day = observation + timedelta(days=1)
                writer.writerow(
                    {
                        "event_name": event.name,
                        "storm_year": event.year,
                        "noaa_storm_id": event.storm_id,
                        "watershed_count": event.watershed_count,
                        "window_type": window,
                        "window_start_utc": start.isoformat(),
                        "window_end_utc_inclusive": end.isoformat(),
                        "core_start_utc": event.core_start.isoformat(),
                        "core_end_utc_inclusive": event.core_end.isoformat(),
                        "observation_date_utc": observation.isoformat(),
                        "interval_start_utc": f"{observation.isoformat()}T00:00:00Z",
                        "interval_end_utc_exclusive": f"{next_day.isoformat()}T00:00:00Z",
                        "is_core_event_date": (
                            "yes"
                            if event.core_start <= observation <= event.core_end
                            else "no"
                        ),
                        "product": "PERSIANN-CDR",
                        "product_version": PRODUCT_VERSION,
                        "source_creation_date": remote.creation_date.isoformat(),
                        "source_url": remote.url,
                        "remote_filename": remote.filename,
                        "local_path": str(result.local_path),
                        "status": result.status,
                        "size_bytes": (
                            "" if result.size_bytes is None else result.size_bytes
                        ),
                        "sha256": result.sha256,
                        "retrieved_utc": result.retrieved_utc,
                        "error": result.error,
                    }
                )

    os.replace(temporary, manifest)
    return manifest


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download native daily PERSIANN-CDR NetCDF files for the study "
            "hurricane windows."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Output directory (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--window",
        choices=("expanded", "core"),
        default="expanded",
        help="Date window to download (default: expanded).",
    )
    parser.add_argument(
        "--event",
        action="append",
        metavar="NAME",
        help="Download one event; repeat for multiple events. Default: all.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="Concurrent downloads (default: 4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--retries",
        type=nonnegative_int,
        default=4,
        help="Retries after the first attempt (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing valid file instead of skipping it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print URLs without creating or downloading files.",
    )
    return parser.parse_args()


def choose_events(requested: list[str] | None) -> list[Event]:
    if not requested:
        return list(EVENTS)

    by_name = {event.name.casefold(): event for event in EVENTS}
    selected: list[Event] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for name in requested:
        key = name.casefold()
        if key not in by_name:
            unknown.append(name)
        elif key not in seen:
            selected.append(by_name[key])
            seen.add(key)

    if unknown:
        valid = ", ".join(event.name for event in EVENTS)
        raise ValueError(f"Unknown event(s): {', '.join(unknown)}. Choose from: {valid}")
    return selected


def main() -> int:
    args = parse_args()
    validate_event_table()

    try:
        selected_events = choose_events(args.event)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    memberships: list[tuple[Event, date]] = []
    for event in selected_events:
        start, end = event.bounds(args.window)
        memberships.extend((event, item) for item in inclusive_dates(start, end))

    unique_dates = sorted({item for _, item in memberships})
    years = sorted({item.year for item in unique_dates})
    remote_files: dict[date, RemoteFile] = {}

    print(f"Window: {args.window}")
    print(f"Events: {', '.join(event.name for event in selected_events)}")
    print(f"Event-day memberships: {len(memberships)}")
    print(f"Unique daily files: {len(unique_dates)}")
    print(f"NOAA year indexes: {', '.join(str(year) for year in years)}")

    try:
        for year in years:
            print(f"Reading NOAA index for {year} ...")
            year_files = discover_year(year, args.timeout, args.retries)
            for observation in unique_dates:
                if observation.year == year and observation in year_files:
                    remote_files[observation] = year_files[observation]
    except Exception as exc:
        print(f"ERROR while reading the NOAA archive: {exc}", file=sys.stderr)
        return 1

    missing = [item for item in unique_dates if item not in remote_files]
    if missing:
        print("ERROR: NOAA files were not found for these dates:", file=sys.stderr)
        for item in missing:
            print(f"  {item.isoformat()}", file=sys.stderr)
        print("No downloads were started.", file=sys.stderr)
        return 1

    if args.dry_run:
        for observation in unique_dates:
            remote = remote_files[observation]
            target = (
                args.destination
                / "daily"
                / str(observation.year)
                / remote.filename
            )
            print(f"{observation.isoformat()}  {remote.url}")
            print(f"    -> {target}")
        print("Dry run complete; no files or directories were created.")
        return 0

    results: dict[date, DownloadResult] = {}
    print(f"Destination: {args.destination}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                remote_files[observation],
                args.destination,
                args.timeout,
                args.retries,
                args.overwrite,
            ): observation
            for observation in unique_dates
        }
        for future in as_completed(futures):
            observation = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Defensive: worker should normally return failure.
                remote = remote_files[observation]
                result = DownloadResult(
                    remote=remote,
                    local_path=(
                        args.destination
                        / "daily"
                        / str(observation.year)
                        / remote.filename
                    ),
                    status="failed",
                    size_bytes=None,
                    sha256="",
                    retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )
            results[observation] = result
            size_text = "" if result.size_bytes is None else f" ({result.size_bytes:,} bytes)"
            print(f"[{result.status}] {observation.isoformat()}{size_text}")
            if result.error:
                print(f"    {result.error}", file=sys.stderr)

    manifest = write_manifest(
        args.destination,
        selected_events,
        args.window,
        remote_files,
        results,
    )
    failures = [item for item in results.values() if item.status == "failed"]
    downloaded = sum(item.status == "downloaded" for item in results.values())
    skipped = sum(item.status == "skipped_valid_existing" for item in results.values())

    print(f"Manifest: {manifest}")
    print(
        f"Finished: {downloaded} downloaded, {skipped} valid existing, "
        f"{len(failures)} failed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
