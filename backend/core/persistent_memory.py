"""
Persistent Memory System - Long-term storage and retrieval using vector database
Implements semantic memory with ChromaDB for the AI's experiences and knowledge
"""

import chromadb
from chromadb.config import Settings
import numpy as np
import torch
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import os
from sentence_transformers import SentenceTransformer


class PersistentMemorySystem:
    """
    Long-term memory system using vector embeddings for semantic storage and retrieval.
    Stores experiences, learned patterns, and knowledge in a persistent database.
    """

    def __init__(
        self,
        memory_path: str = "./memory_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.memory_path = memory_path
        os.makedirs(memory_path, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=memory_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create collections for different memory types
        self.episodic_memory = self._get_or_create_collection("episodic_memory")
        self.semantic_memory = self._get_or_create_collection("semantic_memory")
        self.procedural_memory = self._get_or_create_collection("procedural_memory")

        # Embedding model for semantic search
        self.embedding_model = SentenceTransformer(embedding_model)

        # Memory statistics
        self.stats = {
            'total_memories': 0,
            'episodic_count': 0,
            'semantic_count': 0,
            'procedural_count': 0,
            'last_access': None
        }

        self._update_stats()

    def _get_or_create_collection(self, name: str):
        """Get or create a collection in ChromaDB"""
        try:
            return self.client.get_collection(name=name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

    def store_episodic_memory(
        self,
        experience: Dict[str, Any],
        context: str,
        importance: float = 0.5
    ) -> str:
        """
        Store an episodic memory (specific experience/event)

        Args:
            experience: Dictionary containing state, action, reward, etc.
            context: Natural language description of the experience
            importance: How important this memory is (0-1)

        Returns:
            Memory ID
        """
        memory_id = f"ep_{datetime.now().timestamp()}_{np.random.randint(10000)}"

        # Create embedding from context
        embedding = self.embedding_model.encode(context).tolist()

        # Store in episodic memory
        self.episodic_memory.add(
            embeddings=[embedding],
            documents=[context],
            metadatas=[{
                'type': 'episodic',
                'importance': importance,
                'timestamp': datetime.now().isoformat(),
                'experience_data': json.dumps(experience)
            }],
            ids=[memory_id]
        )

        self.stats['episodic_count'] += 1
        self.stats['total_memories'] += 1

        return memory_id

    def store_semantic_memory(
        self,
        knowledge: str,
        category: str,
        confidence: float = 0.8
    ) -> str:
        """
        Store semantic memory (factual knowledge, concepts)

        Args:
            knowledge: The knowledge/fact to store
            category: Category of knowledge
            confidence: Confidence in this knowledge (0-1)

        Returns:
            Memory ID
        """
        memory_id = f"sem_{datetime.now().timestamp()}_{np.random.randint(10000)}"

        embedding = self.embedding_model.encode(knowledge).tolist()

        self.semantic_memory.add(
            embeddings=[embedding],
            documents=[knowledge],
            metadatas=[{
                'type': 'semantic',
                'category': category,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }],
            ids=[memory_id]
        )

        self.stats['semantic_count'] += 1
        self.stats['total_memories'] += 1

        return memory_id

    def store_procedural_memory(
        self,
        skill_name: str,
        procedure: Dict[str, Any],
        success_rate: float = 0.5
    ) -> str:
        """
        Store procedural memory (how to do things, skills)

        Args:
            skill_name: Name of the skill/procedure
            procedure: Steps or neural patterns for the procedure
            success_rate: How successful this procedure has been (0-1)

        Returns:
            Memory ID
        """
        memory_id = f"proc_{datetime.now().timestamp()}_{np.random.randint(10000)}"

        embedding = self.embedding_model.encode(skill_name).tolist()

        self.procedural_memory.add(
            embeddings=[embedding],
            documents=[skill_name],
            metadatas=[{
                'type': 'procedural',
                'success_rate': success_rate,
                'timestamp': datetime.now().isoformat(),
                'procedure_data': json.dumps(procedure)
            }],
            ids=[memory_id]
        )

        self.stats['procedural_count'] += 1
        self.stats['total_memories'] += 1

        return memory_id

    def recall_similar_memories(
        self,
        query: str,
        memory_type: str = "all",
        n_results: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Recall memories similar to the query

        Args:
            query: Query string to search for
            memory_type: Type of memory ('episodic', 'semantic', 'procedural', 'all')
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar memories with metadata
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        results = []
        collections = []

        # Determine which collections to search
        if memory_type == "all":
            collections = [
                ('episodic', self.episodic_memory),
                ('semantic', self.semantic_memory),
                ('procedural', self.procedural_memory)
            ]
        else:
            if memory_type == "episodic":
                collections = [('episodic', self.episodic_memory)]
            elif memory_type == "semantic":
                collections = [('semantic', self.semantic_memory)]
            elif memory_type == "procedural":
                collections = [('procedural', self.procedural_memory)]

        # Query each collection
        for mem_type, collection in collections:
            try:
                query_result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )

                if query_result['ids'] and len(query_result['ids'][0]) > 0:
                    for i, memory_id in enumerate(query_result['ids'][0]):
                        distance = query_result['distances'][0][i]
                        similarity = 1 - distance  # Convert distance to similarity

                        if similarity >= min_similarity:
                            results.append({
                                'id': memory_id,
                                'type': mem_type,
                                'content': query_result['documents'][0][i],
                                'similarity': similarity,
                                'metadata': query_result['metadatas'][0][i]
                            })
            except Exception as e:
                print(f"Error querying {mem_type} memory: {e}")

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        self.stats['last_access'] = datetime.now().isoformat()

        return results[:n_results]

    def consolidate_memories(self, time_threshold_hours: int = 24):
        """
        Consolidate similar memories (like sleep consolidation in humans)
        Merge or strengthen related memories
        """
        # This is a simplified version - in practice, you'd want more sophisticated consolidation
        pass  # TODO: Implement memory consolidation

    def forget_low_importance_memories(self, importance_threshold: float = 0.3):
        """
        Remove memories below importance threshold (memory decay)
        """
        # Query all episodic memories
        all_episodic = self.episodic_memory.get()

        ids_to_delete = []

        if all_episodic['metadatas']:
            for i, metadata in enumerate(all_episodic['metadatas']):
                if metadata.get('importance', 1.0) < importance_threshold:
                    ids_to_delete.append(all_episodic['ids'][i])

        if ids_to_delete:
            self.episodic_memory.delete(ids=ids_to_delete)
            self.stats['episodic_count'] -= len(ids_to_delete)
            self.stats['total_memories'] -= len(ids_to_delete)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        self._update_stats()
        return self.stats

    def _update_stats(self):
        """Update memory statistics"""
        try:
            self.stats['episodic_count'] = self.episodic_memory.count()
            self.stats['semantic_count'] = self.semantic_memory.count()
            self.stats['procedural_count'] = self.procedural_memory.count()
            self.stats['total_memories'] = (
                self.stats['episodic_count'] +
                self.stats['semantic_count'] +
                self.stats['procedural_count']
            )
        except:
            pass

    def export_memories(self, output_path: str):
        """Export all memories to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'episodic': self.episodic_memory.get(),
            'semantic': self.semantic_memory.get(),
            'procedural': self.procedural_memory.get()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def clear_all_memories(self):
        """Clear all memories (use with caution!)"""
        self.client.reset()
        self.episodic_memory = self._get_or_create_collection("episodic_memory")
        self.semantic_memory = self._get_or_create_collection("semantic_memory")
        self.procedural_memory = self._get_or_create_collection("procedural_memory")
        self._update_stats()


class WorkingMemory:
    """
    Working memory - short-term, currently active information
    Complements the persistent long-term memory
    """

    def __init__(self, capacity: int = 7):  # Miller's Law: 7±2 items
        self.capacity = capacity
        self.buffer = []
        self.attention_weights = []

    def add(self, item: Any, attention: float = 1.0):
        """Add item to working memory with attention weight"""
        self.buffer.append(item)
        self.attention_weights.append(attention)

        # Remove least attended items if over capacity
        if len(self.buffer) > self.capacity:
            min_idx = np.argmin(self.attention_weights)
            self.buffer.pop(min_idx)
            self.attention_weights.pop(min_idx)

    def get_attended_items(self, top_k: int = 3) -> List[Any]:
        """Get the most attended items"""
        if not self.buffer:
            return []

        indices = np.argsort(self.attention_weights)[-top_k:]
        return [self.buffer[i] for i in indices]

    def clear(self):
        """Clear working memory"""
        self.buffer = []
        self.attention_weights = []

    def get_state(self) -> Dict:
        """Get current working memory state"""
        return {
            'capacity': self.capacity,
            'current_size': len(self.buffer),
            'items': [
                {'item': item, 'attention': att}
                for item, att in zip(self.buffer, self.attention_weights)
            ]
        }
