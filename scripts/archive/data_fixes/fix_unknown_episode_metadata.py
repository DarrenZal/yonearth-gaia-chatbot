#!/usr/bin/env python3
"""
Fix vectors with episode_number='unknown' in Pinecone

Identified episodes:
- Episode 137 (125 chunks): Georgia Kelly, Praxis Peace Institute
- Episode 171 (182 chunks): Cynthia James Stewart, James Ranch Grill

This script updates the metadata for these vectors to have correct episode info.
"""

import os
import json
import logging
from pinecone import Pinecone
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv('/home/claudeuser/yonearth-gaia-chatbot/.env')

# Episode metadata to fix
EPISODE_FIXES = {
    125: {  # chunk_total identifies episode 137
        'episode_number': '137',
        'title': 'Episode 137 – Georgia Kelly, Founder, Praxis Peace Institute; on the Mondragon Cooperatives',
        'guest_name': 'Georgia Kelly',
        'subtitle': 'About the Mondragon Cooperatives',
        'url': 'https://yonearth.org/podcast/episode-137-georgia-kelly-founder-praxis-peace-institute-on-the-mondragon-cooperatives/',
        'publish_date': 'June 28, 2023',
    },
    182: {  # chunk_total identifies episode 171
        'episode_number': '171',
        'title': 'Episode 171 – Cynthia James Stewart, Regenerative Burgers & Fries at James Ranch Grill',
        'guest_name': 'Cynthia James Stewart',
        'subtitle': 'Regenerative Fare at James Ranch Grill',
        'url': 'https://yonearth.org/podcast/episode-171-cynthia-james-stewart-regenerative-burgers-fries-at-james-ranch-grill/',
        'publish_date': '',  # Unknown
    }
}


def fix_unknown_metadata():
    """Fix vectors with unknown episode metadata"""

    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index = pc.Index('yonearth-episodes')

    # Get all vectors with unknown episode_number
    logger.info("Fetching vectors with episode_number='unknown'...")

    dummy_vector = [0.0] * 1536

    for chunk_total, fix_data in EPISODE_FIXES.items():
        logger.info(f"\nProcessing episode {fix_data['episode_number']} (chunk_total={chunk_total})...")

        # Query for vectors matching this episode
        results = index.query(
            vector=dummy_vector,
            top_k=300,  # More than max chunks
            include_metadata=True,
            filter={
                "$and": [
                    {"episode_number": {"$eq": "unknown"}},
                    {"chunk_total": {"$eq": chunk_total}}
                ]
            }
        )

        vectors_to_update = results['matches']
        logger.info(f"  Found {len(vectors_to_update)} vectors to update")

        if not vectors_to_update:
            logger.warning(f"  No vectors found for chunk_total={chunk_total}")
            continue

        # Update each vector's metadata
        updated_count = 0
        for match in vectors_to_update:
            vector_id = match['id']
            current_metadata = match.get('metadata', {})

            # Update metadata with correct episode info
            new_metadata = current_metadata.copy()
            new_metadata.update({
                'episode_number': fix_data['episode_number'],
                'title': fix_data['title'],
                'guest_name': fix_data['guest_name'],
                'subtitle': fix_data['subtitle'],
                'url': fix_data['url'],
                'publish_date': fix_data['publish_date'],
            })

            try:
                # Update the vector with new metadata
                # Pinecone update requires the vector values, so we fetch and re-upsert
                fetch_result = index.fetch(ids=[vector_id])
                if vector_id in fetch_result.vectors:
                    vector_data = fetch_result.vectors[vector_id]

                    # Upsert with updated metadata
                    index.upsert(vectors=[{
                        'id': vector_id,
                        'values': vector_data.values,
                        'metadata': new_metadata
                    }])

                    updated_count += 1

                    if updated_count % 50 == 0:
                        logger.info(f"    Updated {updated_count} vectors...")

            except Exception as e:
                logger.error(f"  Error updating vector {vector_id}: {e}")
                continue

        logger.info(f"  ✅ Updated {updated_count} vectors for episode {fix_data['episode_number']}")

    logger.info("\n✅ Metadata fix complete!")

    # Verify the fix
    logger.info("\nVerifying fix...")
    results = index.query(
        vector=dummy_vector,
        top_k=10,
        include_metadata=False,
        filter={"episode_number": {"$eq": "unknown"}}
    )

    remaining = len(results['matches'])
    if remaining == 0:
        logger.info("✅ No more vectors with episode_number='unknown'")
    else:
        logger.warning(f"⚠️  Still {remaining} vectors with episode_number='unknown'")


if __name__ == '__main__':
    fix_unknown_metadata()
