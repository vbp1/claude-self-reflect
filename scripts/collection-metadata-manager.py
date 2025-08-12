#\!/usr/bin/env python3
"""Collection metadata management for Qdrant - Final Implementation."""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
METADATA_COLLECTION = "collection_metadata_local"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))

@dataclass
class CollectionMetadata:
    """Metadata structure for collections."""
    collection_name: str
    project_path: str
    project_name: str
    created_at: str
    description: str
    conversation_count: int
    last_updated: str
    tags: List[str]
    embedding_model: str
    is_active: bool = True
    notes: str = ""

class CollectionMetadataManager:
    """Manager for collection metadata operations."""
    
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self._ensure_metadata_collection()
    
    def _ensure_metadata_collection(self):
        """Ensure metadata collection exists."""
        collections = self.client.get_collections()
        if not any(c.name == METADATA_COLLECTION for c in collections.collections):
            print(f"📦 Creating metadata collection: {METADATA_COLLECTION}")
            self.client.create_collection(
                collection_name=METADATA_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
    
    def _project_path_to_name(self, project_path: str) -> str:
        """Convert project path to human-readable name."""
        if project_path == "unknown":
            return "Unknown Project"
        
        # Convert -home-user-project to user-project
        path_parts = project_path.replace('-home-', '/home/').split('/')
        return path_parts[-1] if path_parts else project_path
    
    def store_metadata(self, metadata: CollectionMetadata) -> bool:
        """Store collection metadata."""
        try:
            # Generate consistent ID based on collection name hash
            metadata_id = abs(hash(metadata.collection_name)) % (10**9)
            
            metadata_point = PointStruct(
                id=metadata_id,
                vector=[0.1] * VECTOR_SIZE,  # Dummy vector for metadata storage
                payload=asdict(metadata)
            )
            
            self.client.upsert(
                collection_name=METADATA_COLLECTION,
                points=[metadata_point]
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing metadata for {metadata.collection_name}: {e}")
            return False
    
    def get_metadata(self, collection_name: str) -> Optional[CollectionMetadata]:
        """Get metadata for a specific collection."""
        try:
            result = self.client.scroll(
                collection_name=METADATA_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="collection_name",
                            match=MatchValue(value=collection_name)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            if result[0] and result[0][0].payload:
                payload = result[0][0].payload
                # Ensure all required fields are present
                if all(key in payload for key in ['collection_name', 'project_path', 'project_name', 'created_at', 'description', 'conversation_count', 'last_updated', 'tags', 'embedding_model']):
                    return CollectionMetadata(**payload)
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting metadata for {collection_name}: {e}")
            return None
    
    def list_all_metadata(self) -> List[CollectionMetadata]:
        """List metadata for all collections."""
        try:
            result = self.client.scroll(
                collection_name=METADATA_COLLECTION,
                limit=1000,
                with_payload=True
            )
            
            metadata_list = []
            for point in result[0]:
                if point.payload and all(key in point.payload for key in ['collection_name', 'project_path', 'project_name', 'created_at', 'description', 'conversation_count', 'last_updated', 'tags', 'embedding_model']):
                    metadata_list.append(CollectionMetadata(**point.payload))
            
            # Sort by project name for better readability
            metadata_list.sort(key=lambda x: x.project_name)
            return metadata_list
            
        except Exception as e:
            print(f"❌ Error listing metadata: {e}")
            return []
    
    def update_metadata(self, collection_name: str, updates: Dict[str, Any]) -> bool:
        """Update existing metadata."""
        metadata = self.get_metadata(collection_name)
        if not metadata:
            print(f"❌ Metadata for collection {collection_name} not found")
            return False
        
        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # Update timestamp
        metadata.last_updated = datetime.now().isoformat()
        
        return self.store_metadata(metadata)
    
    def delete_metadata(self, collection_name: str) -> bool:
        """Delete metadata for a collection."""
        try:
            metadata_id = abs(hash(collection_name)) % (10**9)
            self.client.delete(
                collection_name=METADATA_COLLECTION,
                points_selector=[metadata_id]
            )
            return True
            
        except Exception as e:
            print(f"❌ Error deleting metadata for {collection_name}: {e}")
            return False
    
    def search_by_tag(self, tag: str) -> List[CollectionMetadata]:
        """Search collections by tag."""
        try:
            result = self.client.scroll(
                collection_name=METADATA_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="tags",
                            match=MatchValue(value=tag)
                        )
                    ]
                ),
                limit=100,
                with_payload=True
            )
            
            metadata_list = []
            for point in result[0]:
                if point.payload and all(key in point.payload for key in ['collection_name', 'project_path', 'project_name', 'created_at', 'description', 'conversation_count', 'last_updated', 'tags', 'embedding_model']):
                    metadata_list.append(CollectionMetadata(**point.payload))
            
            return metadata_list
            
        except Exception as e:
            print(f"❌ Error searching by tag {tag}: {e}")
            return []
    
    def auto_discover_metadata(self) -> List[CollectionMetadata]:
        """Auto-discover metadata from existing collections."""
        print("🔍 Auto-discovering metadata for existing collections...")
        
        collections = self.client.get_collections()
        discovered_metadata = []
        
        for collection in collections.collections:
            # Skip non-local collections and metadata collection itself
            if not collection.name.endswith("_local") or collection.name == METADATA_COLLECTION:
                continue
            
            print(f"   Analyzing: {collection.name}")
            
            # Check if metadata already exists
            existing = self.get_metadata(collection.name)
            if existing:
                print("     ⏭ Metadata already exists, skipping")
                continue
            
            try:
                # Get collection info
                info = self.client.get_collection(collection.name)
                
                # Try to extract project info from sample points
                project_path = "unknown"
                project_name = "Unknown Project"
                
                if info.points_count and info.points_count > 0:
                    sample = self.client.scroll(
                        collection_name=collection.name,
                        limit=1,
                        with_payload=True
                    )
                    
                    if sample[0] and sample[0][0].payload:
                        payload = sample[0][0].payload
                        if 'project' in payload:
                            project_path = payload['project']
                        elif 'file_path' in payload:
                            # Extract project from file path
                            file_path = payload['file_path']
                            project_path = '/'.join(file_path.split('/')[:-1])
                        
                        project_name = self._project_path_to_name(project_path)
                
                # Create metadata
                metadata = CollectionMetadata(
                    collection_name=collection.name,
                    project_path=project_path,
                    project_name=project_name,
                    created_at=datetime.now().isoformat(),
                    description=f"Auto-discovered collection for {project_name}",
                    conversation_count=info.points_count or 0,
                    last_updated=datetime.now().isoformat(),
                    tags=["auto-discovered"],
                    embedding_model="local-fastembed",
                    is_active=True
                )
                
                if self.store_metadata(metadata):
                    discovered_metadata.append(metadata)
                    print("     ✅ Metadata stored")
                
            except Exception as e:
                print(f"     ❌ Error analyzing collection {collection.name}: {e}")
        
        return discovered_metadata
    
    def update_conversation_counts(self):
        """Update conversation counts for all collections."""
        print("🔄 Updating conversation counts...")
        
        metadata_list = self.list_all_metadata()
        updated = 0
        
        for metadata in metadata_list:
            try:
                info = self.client.get_collection(metadata.collection_name)
                if info.points_count != metadata.conversation_count:
                    self.update_metadata(metadata.collection_name, {
                        'conversation_count': info.points_count
                    })
                    print(f"   Updated {metadata.collection_name}: {metadata.conversation_count} -> {info.points_count}")
                    updated += 1
            except Exception as e:
                print(f"   ❌ Error updating {metadata.collection_name}: {e}")
        
        print(f"✅ Updated {updated} collections")

def main():
    """Main CLI interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("🔧 Collection Metadata Manager")
        print("=" * 50)
        print("Usage:")
        print("  python collection-metadata-manager.py discover    - Auto-discover metadata")
        print("  python collection-metadata-manager.py list        - List all metadata")
        print("  python collection-metadata-manager.py get <name>  - Get specific metadata")
        print("  python collection-metadata-manager.py update      - Update conversation counts")
        print("  python collection-metadata-manager.py tag <tag>   - Search by tag")
        return
    
    manager = CollectionMetadataManager()
    command = sys.argv[1]
    
    if command == "discover":
        discovered = manager.auto_discover_metadata()
        print(f"\n🎉 Discovered and stored metadata for {len(discovered)} collections")
        
    elif command == "list":
        metadata_list = manager.list_all_metadata()
        print(f"\n📋 COLLECTION METADATA ({len(metadata_list)} collections)")
        print("=" * 80)
        
        for metadata in metadata_list:
            status = "🟢" if metadata.is_active else "🔴"
            tags_str = ", ".join(metadata.tags) if metadata.tags else ""
            
            print(f"{status} {metadata.collection_name}")
            print(f"   Project: {metadata.project_name}")
            print(f"   Path: {metadata.project_path}")
            print(f"   Conversations: {metadata.conversation_count}")
            print(f"   Model: {metadata.embedding_model}")
            print(f"   Tags: {tags_str}")
            if metadata.description:
                print(f"   Description: {metadata.description}")
            print()
            
    elif command == "get" and len(sys.argv) > 2:
        collection_name = sys.argv[2]
        metadata = manager.get_metadata(collection_name)
        
        if metadata:
            print(f"📄 METADATA FOR {collection_name}")
            print("=" * 50)
            print(json.dumps(asdict(metadata), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Metadata for collection {collection_name} not found")
    
    elif command == "update":
        manager.update_conversation_counts()
    
    elif command == "tag" and len(sys.argv) > 2:
        tag = sys.argv[2]
        results = manager.search_by_tag(tag)
        print(f"\n🏷 COLLECTIONS WITH TAG '{tag}' ({len(results)} found)")
        print("=" * 50)
        
        for metadata in results:
            print(f"• {metadata.collection_name} - {metadata.project_name}")
    
    else:
        print("❌ Unknown command or missing arguments")

if __name__ == "__main__":
    main()
