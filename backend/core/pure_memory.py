"""
Pure Developmental Memory System

NO pretrained embeddings.
NO external knowledge.
Concepts emerge through clustering and experience.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
import json
import pickle
import os


class Episode:
    """A single moment in Indi's life"""

    def __init__(
        self,
        timestamp: str,
        sensory_input: Dict[str, Any],
        z_state: np.ndarray,
        teacher_words: Optional[str] = None,
        indi_output: Optional[str] = None,
        correction: Optional[str] = None,
        salience: float = 0.5
    ):
        self.timestamp = timestamp
        self.sensory_input = sensory_input  # Raw pixels, audio, etc.
        self.z_state = z_state  # Latent representation
        self.teacher_words = teacher_words
        self.indi_output = indi_output
        self.correction = correction
        self.salience = salience  # How important/emotional this moment was

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'sensory_input_summary': self._summarize_sensory(),
            'z_state': self.z_state.tolist() if isinstance(self.z_state, np.ndarray) else self.z_state,
            'teacher_words': self.teacher_words,
            'indi_output': self.indi_output,
            'correction': self.correction,
            'salience': self.salience
        }

    def _summarize_sensory(self) -> Dict:
        """Summarize sensory input without storing full raw data"""
        summary = {}
        for key, value in self.sensory_input.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                summary[key] = {
                    'type': str(type(value)),
                    'shape': list(value.shape) if hasattr(value, 'shape') else None
                }
            else:
                summary[key] = value
        return summary


class Concept:
    """
    An emergent concept discovered through clustering.
    Has NO preloaded meaning - only what Indi has experienced.
    """

    def __init__(self, concept_id: int, prototype: np.ndarray):
        self.concept_id = concept_id
        self.prototype = prototype  # Cluster center in z-space
        self.episodes = []  # Episodes that belong to this concept
        self.words = []  # Words teacher associated with this concept
        self.created_at = datetime.now().isoformat()
        self.activation_count = 0

    def add_episode(self, episode_id: int, word: Optional[str] = None):
        """Add an episode to this concept"""
        self.episodes.append(episode_id)
        if word and word not in self.words:
            self.words.append(word)
        self.activation_count += 1

    def get_primary_word(self) -> Optional[str]:
        """Get the most associated word"""
        if not self.words:
            return None
        # Could use frequency counting here
        return self.words[0]

    def to_dict(self) -> Dict:
        return {
            'concept_id': self.concept_id,
            'prototype': self.prototype.tolist(),
            'episodes': self.episodes,
            'words': self.words,
            'created_at': self.created_at,
            'activation_count': self.activation_count
        }


class EpisodicMemory:
    """
    Stores every moment of Indi's life.
    Pure experiential memory with no artificial structure.
    """

    def __init__(self, max_episodes: int = 100000):
        self.episodes = []
        self.max_episodes = max_episodes

    def store(self, episode: Episode) -> int:
        """Store an episode and return its ID"""
        episode_id = len(self.episodes)
        self.episodes.append(episode)

        # Simple capacity management
        if len(self.episodes) > self.max_episodes:
            # Remove lowest salience episodes
            self.episodes.sort(key=lambda e: e.salience)
            self.episodes = self.episodes[int(self.max_episodes * 0.1):]

        return episode_id

    def retrieve(self, episode_id: int) -> Optional[Episode]:
        """Retrieve a specific episode"""
        if 0 <= episode_id < len(self.episodes):
            return self.episodes[episode_id]
        return None

    def retrieve_by_salience(self, top_k: int = 10) -> List[Episode]:
        """Retrieve most salient episodes"""
        sorted_episodes = sorted(self.episodes, key=lambda e: e.salience, reverse=True)
        return sorted_episodes[:top_k]

    def get_recent(self, n: int = 10) -> List[Episode]:
        """Get most recent episodes"""
        return self.episodes[-n:]

    def count(self) -> int:
        return len(self.episodes)


class SemanticMemory:
    """
    Emergent concepts discovered through clustering.
    NO preloaded categories. Pure discovery.
    """

    def __init__(self, n_clusters: int = 50, z_dim: int = 64):
        self.n_clusters = n_clusters
        self.z_dim = z_dim

        # Clustering algorithm (incremental learning)
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=32,
            random_state=42
        )

        self.concepts: Dict[int, Concept] = {}
        self.fitted = False

    def update_clusters(self, z_states: np.ndarray):
        """Update clusters with new experiences"""
        if len(z_states.shape) == 1:
            z_states = z_states.reshape(1, -1)

        if not self.fitted and len(z_states) >= self.n_clusters:
            # Initial fit
            self.clusterer.fit(z_states)
            self.fitted = True
            self._update_concepts()
        elif self.fitted:
            # Incremental update
            self.clusterer.partial_fit(z_states)
            self._update_concepts()

    def _update_concepts(self):
        """Update concept prototypes from cluster centers"""
        for cluster_id, center in enumerate(self.clusterer.cluster_centers_):
            if cluster_id not in self.concepts:
                self.concepts[cluster_id] = Concept(
                    concept_id=cluster_id,
                    prototype=center
                )
            else:
                # Update prototype
                self.concepts[cluster_id].prototype = center

    def find_nearest_concept(self, z: np.ndarray) -> Tuple[int, float]:
        """
        Find the nearest concept to a given z-vector.

        Returns:
            (concept_id, distance)
        """
        if not self.fitted:
            return -1, float('inf')

        if len(z.shape) == 1:
            z = z.reshape(1, -1)

        cluster_id = self.clusterer.predict(z)[0]
        center = self.clusterer.cluster_centers_[cluster_id]
        distance = np.linalg.norm(z - center)

        return int(cluster_id), float(distance)

    def associate_word(self, concept_id: int, episode_id: int, word: str):
        """Associate a word with a concept"""
        if concept_id in self.concepts:
            self.concepts[concept_id].add_episode(episode_id, word)

    def get_concept_word(self, concept_id: int) -> Optional[str]:
        """Get the primary word associated with a concept"""
        if concept_id in self.concepts:
            return self.concepts[concept_id].get_primary_word()
        return None

    def get_all_concepts(self) -> List[Concept]:
        """Get all discovered concepts"""
        return list(self.concepts.values())


class SelfConcept:
    """
    Indi's emerging sense of self.
    Created after birth through teaching.
    """

    def __init__(self):
        self.name = None  # Learned through repetition
        self.self_episodes = []  # Episodes where "Indi" was mentioned
        self.birth_time = datetime.now().isoformat()
        self.awareness_level = 0.0

    def hear_name(self, name: str, episode_id: int):
        """Teacher says Indi's name"""
        if self.name is None:
            self.name = name
        self.self_episodes.append(episode_id)
        # Awareness grows with name recognition
        self.awareness_level = min(1.0, len(self.self_episodes) / 100.0)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'birth_time': self.birth_time,
            'self_episodes': self.self_episodes,
            'awareness_level': self.awareness_level
        }


class PureDevelopmentalMemory:
    """
    The complete memory system for pure developmental learning.

    NO pretrained knowledge.
    NO external embeddings.
    Pure experiential memory + emergent concepts.
    """

    def __init__(
        self,
        memory_path: str = "./indi_memory",
        n_clusters: int = 100,
        z_dim: int = 64
    ):
        self.memory_path = memory_path
        os.makedirs(memory_path, exist_ok=True)

        # Memory systems
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory(n_clusters=n_clusters, z_dim=z_dim)
        self.self_concept = SelfConcept()

        # Stats
        self.stats = {
            'total_episodes': 0,
            'concepts_discovered': 0,
            'words_learned': 0,
            'birth_time': datetime.now().isoformat()
        }

    def store_experience(
        self,
        sensory_input: Dict[str, Any],
        z_state: torch.Tensor,
        teacher_words: Optional[str] = None,
        indi_output: Optional[str] = None,
        correction: Optional[str] = None,
        salience: float = 0.5
    ) -> int:
        """
        Store a complete experience in memory.

        Returns:
            episode_id
        """
        # Convert z to numpy
        if isinstance(z_state, torch.Tensor):
            z_np = z_state.detach().cpu().numpy()
            if len(z_np.shape) > 1:
                z_np = z_np[0]  # Take first if batch
        else:
            z_np = z_state

        # Create episode
        episode = Episode(
            timestamp=datetime.now().isoformat(),
            sensory_input=sensory_input,
            z_state=z_np,
            teacher_words=teacher_words,
            indi_output=indi_output,
            correction=correction,
            salience=salience
        )

        # Store in episodic memory
        episode_id = self.episodic.store(episode)

        # Update semantic memory (clustering)
        self.semantic.update_clusters(z_np.reshape(1, -1))

        # Find which concept this belongs to
        concept_id, distance = self.semantic.find_nearest_concept(z_np)

        # Associate word if provided
        if teacher_words and concept_id >= 0:
            # Extract first word (simple parsing)
            first_word = teacher_words.strip().split()[0] if teacher_words.strip() else None
            if first_word:
                self.semantic.associate_word(concept_id, episode_id, first_word.lower())

        # Check if teacher mentioned Indi's name
        if teacher_words and self.self_concept.name:
            if self.self_concept.name.lower() in teacher_words.lower():
                self.self_concept.hear_name(self.self_concept.name, episode_id)
        elif teacher_words and "indi" in teacher_words.lower():
            # Learn name
            self.self_concept.hear_name("Indi", episode_id)

        # Update stats
        self.stats['total_episodes'] = self.episodic.count()
        self.stats['concepts_discovered'] = len(self.semantic.concepts)
        self.stats['words_learned'] = len(set(
            word for concept in self.semantic.concepts.values() for word in concept.words
        ))

        return episode_id

    def recall_similar(self, z_query: torch.Tensor, top_k: int = 5) -> List[Tuple[Episode, float]]:
        """
        Recall episodes similar to a query in z-space.

        Returns:
            List of (episode, similarity_score) tuples
        """
        if isinstance(z_query, torch.Tensor):
            z_query_np = z_query.detach().cpu().numpy()
            if len(z_query_np.shape) > 1:
                z_query_np = z_query_np[0]
        else:
            z_query_np = z_query

        # Calculate distances to all episodes
        similarities = []
        for episode in self.episodic.episodes:
            distance = np.linalg.norm(z_query_np - episode.z_state)
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            similarities.append((episode, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_concept_for_word(self, word: str) -> Optional[int]:
        """Find the concept ID associated with a word"""
        word_lower = word.lower()
        for concept in self.semantic.concepts.values():
            if word_lower in [w.lower() for w in concept.words]:
                return concept.concept_id
        return None

    def get_word_for_z(self, z: torch.Tensor) -> Optional[str]:
        """Get the word associated with a z-vector (if any)"""
        if isinstance(z, torch.Tensor):
            z_np = z.detach().cpu().numpy()
            if len(z_np.shape) > 1:
                z_np = z_np[0]
        else:
            z_np = z

        concept_id, distance = self.semantic.find_nearest_concept(z_np)

        if concept_id >= 0 and distance < 2.0:  # Threshold for "close enough"
            return self.semantic.get_concept_word(concept_id)

        return None

    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            **self.stats,
            'self_concept': self.self_concept.to_dict(),
            'recent_episodes': [e.to_dict() for e in self.episodic.get_recent(5)]
        }

    def save(self):
        """Save memory to disk"""
        save_path = os.path.join(self.memory_path, 'memory_state.pkl')

        state = {
            'episodic_episodes': [e.to_dict() for e in self.episodic.episodes],
            'semantic_concepts': {cid: c.to_dict() for cid, c in self.semantic.concepts.items()},
            'semantic_clusterer': self.semantic.clusterer,
            'self_concept': self.self_concept.to_dict(),
            'stats': self.stats
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Memory saved to {save_path}")

    def load(self):
        """Load memory from disk"""
        save_path = os.path.join(self.memory_path, 'memory_state.pkl')

        if not os.path.exists(save_path):
            print("No saved memory found")
            return

        with open(save_path, 'rb') as f:
            state = pickle.load(f)

        # Reconstruct episodic memory
        # (simplified - in production you'd fully reconstruct Episode objects)
        self.stats = state['stats']

        # Restore semantic clusterer
        self.semantic.clusterer = state['semantic_clusterer']
        self.semantic.fitted = True

        # Restore self concept
        self.self_concept.name = state['self_concept'].get('name')
        self.self_concept.awareness_level = state['self_concept'].get('awareness_level', 0.0)

        print(f"Memory loaded from {save_path}")
