"""Local-only original-report snapshot CLI; never downloads or places orders."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alpharesearch.snapshots import create_event_snapshot, verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('snapshot')
    create.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    create.add_argument('--root')
    verify = sub.add_parser('verify');verify.add_argument('folder')
    args = parser.parse_args()
    if args.command == 'snapshot':
        folder = create_event_snapshot(args.project, args.root, progress=lambda s: print(s, file=sys.stderr, flush=True))
    else:
        folder = Path(args.folder)
    manifest = verify_snapshot(folder)
    print(json.dumps({'folder': str(folder.resolve()), 'snapshot_id': manifest['snapshot_id'],
                      'quality': manifest['quality'], 'limits': manifest['limits']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
