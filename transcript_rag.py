#!/usr/bin/env python3
"""
RAG System for YouTube Transcripts
Ask questions about your video content using local AI

Now uses enhanced version with:
- Chunking with overlap
- Hybrid search (semantic + keyword)
- Query expansion

NOTE: v2 is available with multi-step retrieval and better reasoning!
See: transcript_rag_v2.py and RAG_V2_IMPROVEMENTS.md
"""

# Import enhanced version (v1 - current stable)
from transcript_rag_enhanced import EnhancedTranscriptRAG

# Use enhanced version as the main class (backward compatible)
TranscriptRAG = EnhancedTranscriptRAG

# To use v2 instead, uncomment this:
# from transcript_rag_v2 import TranscriptRAG as TranscriptRAGv2
# TranscriptRAG = TranscriptRAGv2

# The enhanced version has all the same methods, just improved:
# - index_single_transcript() now uses chunking automatically
# - search_transcripts() now uses hybrid search by default
# - ask() now uses query expansion and hybrid search

if __name__ == '__main__':
    # Interactive RAG session
    print("🚀 YouTube Transcript RAG System (Enhanced)")
    print("=" * 60)
    
    rag = TranscriptRAG()
    
    # Check if transcripts are indexed
    try:
        collection = rag.chroma_client.get_collection("transcripts")
        count = collection.count()
        print(f"✅ {count} items already indexed")
    except:
        print("📝 No transcript index found.")
        print("   Run: python reindex_missing_videos.py to index videos")
    
    print("\n" + "=" * 60)
    print("💬 Ask questions about your video transcripts!")
    print("   Examples:")
    print("   - What did I watch about neural networks?")
    print("   - In the LUCA video, what doesn't fit?")
    print("   - What videos discussed climate change?")
    print("\n   Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            rag.ask(question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

