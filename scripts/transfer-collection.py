#!/usr/bin/env python3
"""
Transfer data from one Qdrant collection to another.
"""
import os
import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

def transfer_collection(source_collection: str, target_collection: str):
    """Transfer all data from source to target collection."""
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if source collection exists
    try:
        source_info = client.get_collection(source_collection)
        logger.info(f"Source collection '{source_collection}' found with {source_info.points_count} points")
    except Exception as e:
        logger.error(f"Source collection '{source_collection}' not found: {e}")
        return False
    
    # Create target collection with same configuration as source
    try:
        vector_config = source_info.config.params.vectors
        logger.info(f"Creating target collection '{target_collection}' with vector config: {vector_config}")
        
        client.create_collection(
            collection_name=target_collection,
            vectors_config=vector_config
        )
        logger.info(f"Target collection '{target_collection}' created")
    except Exception as e:
        logger.warning(f"Target collection creation failed (may already exist): {e}")
    
    # Transfer data in batches
    batch_size = 100
    offset = 0
    total_transferred = 0
    
    while True:
        # Get batch of points from source
        try:
            response = client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            points = response[0]  # points
            next_offset = response[1]  # next page offset
            
            if not points:
                break
                
            logger.info(f"Transferring batch of {len(points)} points...")
            
            # Convert Record objects to PointStruct format
            from qdrant_client.models import PointStruct
            point_structs = []
            
            for point in points:
                point_struct = PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload
                )
                point_structs.append(point_struct)
            
            # Insert points into target collection
            client.upsert(
                collection_name=target_collection,
                points=point_structs
            )
            
            total_transferred += len(points)
            logger.info(f"Transferred {total_transferred} points so far")
            
            # Check if we've reached the end
            if next_offset is None:
                break
                
            offset = next_offset
            
        except Exception as e:
            logger.error(f"Error during transfer: {e}")
            return False
    
    logger.info(f"Transfer completed! {total_transferred} points transferred from '{source_collection}' to '{target_collection}'")
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python transfer-collection.py <source_collection> <target_collection>")
        sys.exit(1)
    
    source_collection = sys.argv[1]
    target_collection = sys.argv[2]
    
    logger.info(f"Starting transfer from '{source_collection}' to '{target_collection}'")
    
    success = transfer_collection(source_collection, target_collection)
    
    if success:
        logger.info("Transfer completed successfully!")
        print(f"✅ Successfully transferred data from '{source_collection}' to '{target_collection}'")
    else:
        logger.error("Transfer failed!")
        print(f"❌ Failed to transfer data from '{source_collection}' to '{target_collection}'")
        sys.exit(1)

if __name__ == "__main__":
    main()