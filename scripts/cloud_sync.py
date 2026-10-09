"""
scripts/cloud_sync.py — Production CLI Tool for Multi-Node ML & Trade Data Synchronization.

Usage:
    python scripts/cloud_sync.py --status
    python scripts/cloud_sync.py --full
    python scripts/cloud_sync.py --push
    python scripts/cloud_sync.py --pull
    python scripts/cloud_sync.py --models
    python scripts/cloud_sync.py --retrain
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.cloud_sync import CloudSyncEngine, CloudSyncClient
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Enterprise ML & Trade Data Cloud Synchronizer")
    parser.add_argument("--url", type=str, default=os.getenv("CLOUD_SYNC_URL"), help="Remote Cloud URL (e.g. http://13.234.xx.xx:8000)")
    parser.add_argument("--token", type=str, default=os.getenv("SYNC_AUTH_TOKEN", ""), help="Sync Authorization Token")
    parser.add_argument("--status", action="store_true", help="Check connection status and compare local vs remote event counts")
    parser.add_argument("--push", action="store_true", help="Push local trades and ML events to cloud")
    parser.add_argument("--pull", action="store_true", help="Pull cloud trades and ML events to local DB")
    parser.add_argument("--models", action="store_true", help="Download latest compiled ML models (.joblib) from cloud")
    parser.add_argument("--retrain", action="store_true", help="Trigger model retraining on remote cloud instance")
    parser.add_argument("--full", action="store_true", help="Perform complete two-way synchronization (Push + Pull + Models)")

    args = parser.parse_args()

    engine = CloudSyncEngine()

    print("=" * 65)
    print("      🚀 ENTERPRISE CLOUD ML & DATA SYNCHRONIZER 🚀")
    print("=" * 65)

    local_status = engine.get_sync_status()
    print(f"📁 Local DB: {local_status['db_path']}")
    print(f"📊 Local Records: ML Events={local_status['counts']['ml_events']:,} | Trades={local_status['counts']['trade_log']:,} | PartialTP={local_status['counts']['partial_tp_events']:,}")
    print(f"🧠 Local Models: {local_status['models_count']} compiled models in ml/artifacts/")

    if not args.url:
        print("\n❌ Error: No Cloud URL specified. Provide --url http://<AWS_IP>:8000 or set CLOUD_SYNC_URL in .env")
        sys.exit(1)

    print(f"🌐 Remote Target: {args.url}")
    print("-" * 65)

    client = CloudSyncClient(remote_url=args.url, auth_token=args.token, local_engine=engine)

    # 1. Status Check
    ok, remote_status = client.check_health()
    if not ok:
        print(f"❌ Failed to connect to remote server: {remote_status.get('error')}")
        sys.exit(1)

    print(f"✅ Connected to Remote Server successfully!")
    r_counts = remote_status.get("counts", {})
    print(f"📊 Remote Records: ML Events={r_counts.get('ml_events', 0):,} | Trades={r_counts.get('trade_log', 0):,} | PartialTP={r_counts.get('partial_tp_events', 0):,}")
    print(f"🧠 Remote Models: {remote_status.get('models_count', 0)} compiled models")

    # If only status was requested
    if args.status and not (args.push or args.pull or args.models or args.retrain or args.full):
        print("\n[INFO] Status check complete.")
        return

    # Default to full sync if no specific sub-action selected
    if not (args.push or args.pull or args.models or args.retrain):
        args.full = True

    # 2. Push Events
    if args.push or args.full:
        print("\n[1/3] 📤 Pushing Local Events to Cloud...")
        res = client.push_events_to_cloud()
        if res.get("status") == "success":
            pushed = res.get("pushed", {})
            print(f"  ✓ Exported & sent {pushed.get('ml_events', 0)} ML events, {pushed.get('trade_log', 0)} trades.")
            imp = res.get("remote_result", {}).get("imported", {})
            print(f"  ✓ Remote inserted: {imp.get('ml_events_inserted', 0)} new ML events, {imp.get('trade_log_inserted', 0)} new trades.")
        else:
            print(f"  ✗ Push failed: {res.get('error')}")

    # 3. Pull Events
    if args.pull or args.full:
        print("\n[2/3] 📥 Pulling Cloud Events to Local DB...")
        res = client.pull_events_from_cloud()
        if res.get("status") == "success":
            pulled = res.get("pulled", {})
            print(f"  ✓ Downloaded {pulled.get('ml_events', 0)} ML events, {pulled.get('trade_log', 0)} trades.")
            imp = res.get("imported", {})
            print(f"  ✓ Local inserted: {imp.get('ml_events_inserted', 0)} new ML events, {imp.get('trade_log_inserted', 0)} new trades.")
        else:
            print(f"  ✗ Pull failed: {res.get('error')}")

    # 4. Sync Models
    if args.models or args.full:
        print("\n[3/3] 🧠 Synchronizing Compiled ML Models (.joblib)...")
        res = client.sync_models_from_cloud()
        if res.get("status") == "success":
            dl = res.get("downloaded", [])
            up = res.get("up_to_date", [])
            errs = res.get("errors", [])
            if dl:
                print(f"  ✓ Downloaded & verified {len(dl)} updated models: {dl}")
            if up:
                print(f"  ✓ {len(up)} models already up-to-date: {up}")
            if errs:
                print(f"  ✗ Errors: {errs}")
        else:
            print(f"  ✗ Model sync failed: {res.get('error')}")

    # 5. Remote Retrain Trigger
    if args.retrain:
        print("\n[+] ⚙️ Requesting Remote Cloud Instance to Retrain on Combined Data...")
        res = client.trigger_remote_retrain()
        print(f"  ✓ Remote response: {res}")

    print("\n" + "=" * 65)
    print("✨ SYNCHRONIZATION COMPLETED SUCCESSFULLY! ✨")
    print("=" * 65)


if __name__ == "__main__":
    main()
