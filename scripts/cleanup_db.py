#!/usr/bin/env python3
"""
Cleanup script to remove bad data from the Observatory database.

Removes:
- Records where model = 'unknown'
- Records where error is not null (failed requests)

Usage:
    python cleanup_db.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load from api/.env first (where Supabase secrets are stored)
api_env = Path(__file__).parent.parent / "api" / ".env"
if api_env.exists():
    load_dotenv(api_env)
else:
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: Set SUPABASE_URL and SUPABASE_KEY in api/.env")
        return
    
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("🧹 Cleaning up Observatory database...\n")
    
    # Count before cleanup
    all_records = client.table("metrics").select("id", count="exact").execute()
    total_before = all_records.count or 0
    print(f"📊 Total records before cleanup: {total_before}")
    
    # Find records with unknown model
    unknown_model = client.table("metrics").select("id").eq("model", "unknown").execute()
    unknown_count = len(unknown_model.data)
    print(f"   - Records with model='unknown': {unknown_count}")
    
    # Find records with errors
    error_records = client.table("metrics").select("id").not_.is_("error", "null").execute()
    error_count = len(error_records.data)
    print(f"   - Records with errors: {error_count}")
    
    if unknown_count == 0 and error_count == 0:
        print("\n✅ Database is already clean!")
        return
    
    # Confirm deletion
    print(f"\n⚠️  About to delete {unknown_count + error_count} records.")
    confirm = input("Continue? [y/N]: ").strip().lower()
    
    if confirm != 'y':
        print("❌ Cancelled.")
        return
    
    # Delete unknown model records
    if unknown_count > 0:
        client.table("metrics").delete().eq("model", "unknown").execute()
        print(f"   ✓ Deleted {unknown_count} records with model='unknown'")
    
    # Delete error records
    if error_count > 0:
        client.table("metrics").delete().not_.is_("error", "null").execute()
        print(f"   ✓ Deleted {error_count} records with errors")
    
    # Count after cleanup
    all_records_after = client.table("metrics").select("id", count="exact").execute()
    total_after = all_records_after.count or 0
    
    print(f"\n✅ Cleanup complete!")
    print(f"   Records remaining: {total_after}")
    print(f"   Records deleted: {total_before - total_after}")


if __name__ == "__main__":
    main()

