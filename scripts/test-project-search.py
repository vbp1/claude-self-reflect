#!/usr/bin/env python3
"""Test project search functionality."""

import asyncio
import subprocess
import hashlib
from pathlib import Path
from qdrant_client import AsyncQdrantClient

async def test_project_search():
    """Test how project search works for different directories."""
    client = AsyncQdrantClient(url='http://localhost:6333')
    
    # Test directory
    test_dir = '/home/vbponomarev/xagent'
    
    print(f"Testing project search for: {test_dir}")
    print("=" * 60)
    
    # 1. Get git root
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=test_dir,
            capture_output=True,
            text=True,
            check=True
        )
        git_root_path = result.stdout.strip()
        git_root_name = Path(git_root_path).name
        print(f"Git root path: {git_root_path}")
        print(f"Git root name: {git_root_name}")
    except subprocess.CalledProcessError:
        print("Not a git repository")
        git_root_path = None
        git_root_name = None
    
    # 2. Get metadata
    metadata_collection = 'collection_metadata_local'
    all_metadata = []
    try:
        points, _ = await client.scroll(
            collection_name=metadata_collection,
            limit=1000,
            with_payload=True
        )
        all_metadata = [p.payload for p in points]
        print(f"\nFound {len(all_metadata)} metadata entries")
    except Exception as e:
        print(f"Error getting metadata: {e}")
    
    # 3. Test metadata search
    if git_root_path and all_metadata:
        # Convert git path to watcher format
        project_path = git_root_path.replace('/', '-')
        if project_path.startswith('-'):
            project_path = project_path[1:]
        project_path = '-' + project_path
        
        print(f"\nConverted project path: {project_path}")
        project_parts = project_path.split('-')
        print(f"Project parts: {project_parts}")
        
        # Find matching collections
        matching_collections = []
        for metadata in all_metadata:
            stored_path = metadata.get("project_path", "")
            stored_parts = stored_path.split('-')
            
            print(f"\nChecking: {metadata['collection_name']}")
            print(f"  Stored path: {stored_path}")
            print(f"  Stored parts: {stored_parts}")
            
            # Check if stored_path is a subpath of project_path
            if len(stored_parts) >= len(project_parts):
                if stored_parts[:len(project_parts)] == project_parts:
                    print("  → MATCH!")
                    matching_collections.append(metadata["collection_name"])
                else:
                    print("  → No match (parts don't match)")
            else:
                print("  → No match (stored path shorter)")
        
        print(f"\nMatching collections: {matching_collections}")
    
    # 4. Test hash-based search (fallback)
    if git_root_name:
        project_hash = hashlib.md5(git_root_name.encode()).hexdigest()[:8]
        print("\nHash-based search:")
        print(f"Project name: {git_root_name}")
        print(f"Project hash: {project_hash}")
        
        # Get all collections
        collections = await client.get_collections()
        hash_collections = [
            c.name for c in collections.collections
            if c.name.startswith(f"conv_{project_hash}_")
        ]
        print(f"Collections by hash: {hash_collections}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_project_search())