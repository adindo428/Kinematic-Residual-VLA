from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from .config import DEFAULT, file_hash, fingerprint, load_config
from .storage import atomic_json, jsonable


def parser():
    p = argparse.ArgumentParser(description="Franka hover-tracking collector and DOM v2.1 export")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect-scene", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT)
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        command.add_argument("--allow-version-mismatch", action="store_true", help="Inspection/smoke only; never training")
        if name == "inspect-scene":
            command.add_argument("--assemble", action="store_true", help="Launch Isaac Lab to choose a scene/table layout")
        else:
            command.add_argument("--paired-modes", action="store_true", help="Record every trajectory at both -45 and +45 degrees")
            command.add_argument("--smoke-suite", action="store_true", help="Run five disposable checks in one simulator session")
    command = sub.add_parser("export-dom")
    command.add_argument("--run", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--allow-smoke", action="store_true")
    command = sub.add_parser("validate")
    command.add_argument("--dataset", required=True, type=Path)
    command.add_argument("--loader-check", action="store_true")
    command.add_argument("--repo-id", default="local/dom-tracking")
    command.add_argument("--report", type=Path, help="Write report OUTSIDE dataset to preserve checksums")
    command = sub.add_parser("upload")
    command.add_argument("--dataset", required=True, type=Path)
    command.add_argument("--repo-id", required=True)
    command.add_argument("--validation-report", required=True, type=Path)
    command.add_argument("--public", action="store_true")
    return p


def launch_isaac(cfg, allow_mismatch, inspection):
    try:
        package_version = importlib.metadata.version("isaaclab")
        spec = importlib.util.find_spec("isaaclab")
        version_files = [p/"VERSION" for p in Path(spec.origin).parents if (p/"VERSION").is_file()]
        version = version_files[0].read_text().strip() if version_files else "unknown-release"
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise ValueError(f"Isaac Lab runtime unavailable ({exc}). Use a Python 3.11 collection environment configured for Isaac Sim 5.1.") from exc
    if version != "2.3.2":
        if not allow_mismatch or (not inspection and cfg["collection"]["purpose"] == "training"):
            raise ValueError(f"Expected Isaac Lab 2.3.2, found {version}. --allow-version-mismatch is limited to inspection/smoke runs.")
    launcher = AppLauncher(headless=cfg["scene"]["headless"], enable_cameras=True, device=cfg["scene"]["device"])
    software = {"isaaclab": version, "isaaclab_package": package_version}
    for name in ("isaacsim", "torch", "numpy", "pyarrow", "av"):
        try:
            software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            software[name] = "bundled"
    from . import __version__
    software["collector"] = __version__
    software["collector_source_sha256"] = fingerprint({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    return launcher.app, software


def run(args):
    if args.command in ("inspect-scene", "collect"):
        from .assets import audit_assets, check_layout
        cfg = load_config(args.config, args.set)
        if args.command == "collect" and args.smoke_suite:
            cfg["collection"]["purpose"] = "smoke"
        audit = audit_assets(cfg)
        if args.command == "inspect-scene" and not args.assemble:
            return {"counts": audit["counts"], "manifest": cfg["assets"]["manifest"],
                    "next": "inspect-scene --assemble through Isaac Lab"}
        app, software = launch_isaac(cfg, args.allow_version_mismatch, args.command == "inspect-scene")
        try:
            from .isaac_backend import IsaacBackend, assemble_layout, dependency_check
            if args.command == "inspect-scene":
                result = assemble_layout(cfg, audit)
            else:
                layout = check_layout(cfg, audit)
                if args.paired_modes:
                    layout['verified_dependencies'] = {k: dependency_check(layout[k]) for k in ('scene_usd','object_usd')}
                backend = IsaacBackend(cfg, layout)
                try:
                    from .collect import collect, smoke_suite
                    if args.paired_modes:
                        if args.smoke_suite:
                            raise ValueError('Choose either paired modes or the legacy smoke suite')
                        from .paired import collect_paired
                        result = collect_paired(cfg, backend, layout, software)
                    else:
                        result = smoke_suite(cfg, backend, layout, software) if args.smoke_suite else collect(cfg, backend, layout, software)
                finally:
                    backend.close()
            # Kit's fast shutdown can exit the process. Emit results BEFORE close.
            print(json.dumps(jsonable(result), indent=2, allow_nan=False), flush=True)
        except BaseException:
            import traceback
            traceback.print_exc()
            import sys
            sys.stderr.flush()
            app._app.post_quit(1)
            raise
        finally:
            app.close()
        return None
    if args.command == "export-dom":
        from .export import export_dom
        return export_dom(args.run, args.output, args.allow_smoke)
    if args.command == "validate":
        from .validate import validate_dataset
        if args.report and args.report.resolve().is_relative_to(args.dataset.resolve()):
            raise ValueError("Validation report must be outside the checksummed dataset")
        result = validate_dataset(args.dataset, args.loader_check, args.repo_id)
        if args.report:
            atomic_json(args.report, result)
        return result
    if args.command == "upload":
        from .hub import upload_dataset
        return upload_dataset(args.dataset, args.repo_id, args.validation_report, args.public)


def main():
    args = parser().parse_args()
    try:
        result = run(args)
        if result is not None:
            print(json.dumps(jsonable(result), indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
