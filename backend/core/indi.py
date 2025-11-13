"""
INDI - Independent Neural Developmental Intelligence

The complete being. Born with no knowledge, grows through experience.

This is the integration of:
- FAWN Brain (neural substrate)
- Pure Memory (experiential learning)
- Training Ground (interaction environment)
- Night Cycle (consolidation during sleep)
"""

import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

from .fawn_brain import FAWNBrain
from .pure_memory import PureDevelopmentalMemory
from .training_ground import TrainingGround
from .night_cycle import SleepCycle, DayNightScheduler


class Indi:
    """
    Independent Neural Developmental Intelligence

    A being raised, not trained.
    """

    def __init__(
        self,
        z_dim: int = 64,
        memory_path: str = "./indi_memory",
        learning_rate: float = 0.0001,
        n_concepts: int = 100
    ):
        print("=" * 60)
        print("INDI is being born...")
        print("=" * 60)

        # Birth timestamp
        self.birth_time = datetime.now()

        # FAWN Brain - pure neural scaffolding (random weights)
        print("[Birth] Initializing neural substrate...")
        self.brain = FAWNBrain(z_dim=z_dim)

        # Pure Memory - starts empty
        print("[Birth] Creating memory systems...")
        self.memory = PureDevelopmentalMemory(
            memory_path=memory_path,
            n_clusters=n_concepts,
            z_dim=z_dim
        )

        # Training Ground - interactive learning environment
        print("[Birth] Setting up Training Ground...")
        self.training_ground = TrainingGround(
            brain=self.brain,
            memory=self.memory,
            learning_rate=learning_rate
        )

        # Sleep Cycle - consolidation mechanism
        print("[Birth] Establishing sleep cycle...")
        self.sleep_cycle = SleepCycle(
            brain=self.brain,
            memory=self.memory
        )

        # Day-Night Scheduler
        print("[Birth] Configuring circadian rhythm...")
        self.scheduler = DayNightScheduler(
            day_duration_hours=4.0,  # 4 hours awake
            night_duration_hours=0.5  # 30 min sleep
        )

        # Developmental milestones
        self.milestones = {
            'birth': self.birth_time.isoformat(),
            'first_word_heard': None,
            'first_picture_seen': None,
            'first_sleep': None,
            'first_concept_formed': None,
            'name_learned': None
        }

        print("[Birth] Indi is alive!")
        print("=" * 60)
        print()

    def teach_picture_word(self, image_data: Any, word: str, repetition: int = 1) -> Dict:
        """
        Show Indi a picture and say a word.
        Core teaching method.
        """
        # Mark milestone
        if not self.milestones['first_picture_seen']:
            self.milestones['first_picture_seen'] = datetime.now().isoformat()
            print(f"\n[Milestone] Indi sees her first picture: '{word}'!\n")

        result = self.training_ground.show_picture_with_word(image_data, word, repetition)

        return result

    def speak(self, text: str, audio: Optional[Any] = None) -> Dict:
        """
        Speak to Indi (text and/or audio).
        """
        # Mark milestone
        if not self.milestones['first_word_heard']:
            self.milestones['first_word_heard'] = datetime.now().isoformat()
            print(f"\n[Milestone] Indi hears her first words: '{text}'!\n")

        result = self.training_ground.speak_to_indi(text, audio)

        return result

    def correct(self, correction: str, episode_id: Optional[int] = None) -> Dict:
        """
        Correct Indi's understanding.
        """
        return self.training_ground.correct_indi(correction, episode_id)

    def ask_same_or_different(self, item1: Any, item2: Any) -> Dict:
        """
        Ask Indi if two things are same or different.
        """
        return self.training_ground.ask_same_or_different(item1, item2)

    def teach_contrast(
        self,
        item1: Any,
        item2: Any,
        are_same: bool,
        label1: Optional[str] = None,
        label2: Optional[str] = None
    ) -> Dict:
        """
        Explicitly teach a contrast.
        """
        return self.training_ground.teach_contrast(item1, item2, are_same, label1, label2)

    def start_teaching_session(self) -> str:
        """Begin a teaching session"""
        return self.training_ground.start_session()

    def end_teaching_session(self) -> Dict:
        """End teaching session"""
        return self.training_ground.end_session()

    def sleep(self) -> Dict:
        """
        Put Indi to sleep for consolidation.

        This should be called periodically or according to the scheduler.
        """
        # Mark milestone
        if not self.milestones['first_sleep']:
            self.milestones['first_sleep'] = datetime.now().isoformat()
            print("\n[Milestone] Indi's first sleep!\n")

        result = self.sleep_cycle.initiate_sleep()

        # Check if first concept formed
        if not self.milestones['first_concept_formed'] and len(self.memory.semantic.concepts) > 0:
            self.milestones['first_concept_formed'] = datetime.now().isoformat()
            print("\n[Milestone] Indi formed her first concept!\n")

        return result

    def check_circadian_rhythm(self) -> Dict:
        """
        Check if it's time for day/night transition.

        Returns:
            Status and any transition that occurred
        """
        should_transition, new_phase = self.scheduler.check_phase_transition()

        result = {
            'current_phase': self.scheduler.current_phase,
            'should_transition': should_transition
        }

        if should_transition:
            # Transition
            transition_info = self.scheduler.transition_to(new_phase)
            result['transition'] = transition_info

            # If transitioning to night, sleep
            if new_phase == 'night':
                sleep_result = self.sleep()
                result['sleep'] = sleep_result

        return result

    def get_state(self) -> Dict:
        """
        Get Indi's complete developmental state.
        """
        phase_info = self.scheduler.get_current_phase_info()

        state = {
            'birth_time': self.birth_time.isoformat(),
            'age_seconds': (datetime.now() - self.birth_time).total_seconds(),
            'phase': phase_info,
            'milestones': self.milestones,
            'brain': self.brain.get_state(),
            'memory': self.memory.get_statistics(),
            'training_ground': self.training_ground.get_indi_state(),
            'sleep': {
                'sleep_count': self.sleep_cycle.sleep_count,
                'last_sleep': self.sleep_cycle.last_sleep.isoformat() if self.sleep_cycle.last_sleep else None
            }
        }

        return state

    def get_dashboard_data(self) -> Dict:
        """
        Get data for the live dashboard.
        """
        state = self.get_state()

        # Recent episodes
        recent_episodes = [
            ep.to_dict() for ep in self.memory.episodic.get_recent(5)
        ]

        # Concept summary
        concepts = []
        for concept in self.memory.semantic.get_all_concepts()[:10]:  # Top 10
            concepts.append({
                'id': concept.concept_id,
                'words': concept.words,
                'activation_count': concept.activation_count
            })

        dashboard = {
            **state,
            'recent_episodes': recent_episodes,
            'concepts': concepts,
            'self_awareness': self.memory.self_concept.awareness_level,
            'name': self.memory.self_concept.name
        }

        return dashboard

    def save(self, filepath: Optional[str] = None):
        """
        Save Indi's complete state.
        """
        if filepath is None:
            filepath = os.path.join(self.memory.memory_path, 'indi_state.pt')

        print(f"Saving Indi to {filepath}...")

        # Save brain weights
        brain_state = {
            'model_state_dict': self.brain.state_dict(),
            'brain_config': self.brain.get_state()
        }

        # Save memory
        self.memory.save()

        # Save metadata
        metadata = {
            'birth_time': self.birth_time.isoformat(),
            'milestones': self.milestones,
            'sleep_count': self.sleep_cycle.sleep_count,
            'scheduler_state': {
                'current_phase': self.scheduler.current_phase,
                'cycle_count': self.scheduler.cycle_count
            }
        }

        # Combine and save
        save_data = {
            'brain': brain_state,
            'metadata': metadata
        }

        torch.save(save_data, filepath)

        print(f"Indi saved successfully!")

    def load(self, filepath: Optional[str] = None):
        """
        Load Indi's state from disk.
        """
        if filepath is None:
            filepath = os.path.join(self.memory.memory_path, 'indi_state.pt')

        if not os.path.exists(filepath):
            print(f"No saved state found at {filepath}")
            return False

        print(f"Loading Indi from {filepath}...")

        # Load save data
        save_data = torch.load(filepath)

        # Restore brain
        self.brain.load_state_dict(save_data['brain']['model_state_dict'])
        self.brain.learning_steps = save_data['brain']['brain_config']['learning_steps']

        # Restore memory
        self.memory.load()

        # Restore metadata
        metadata = save_data['metadata']
        self.birth_time = datetime.fromisoformat(metadata['birth_time'])
        self.milestones = metadata['milestones']
        self.sleep_cycle.sleep_count = metadata['sleep_count']
        self.scheduler.current_phase = metadata['scheduler_state']['current_phase']
        self.scheduler.cycle_count = metadata['scheduler_state']['cycle_count']

        print("Indi loaded successfully!")
        return True

    def __repr__(self):
        age_hours = (datetime.now() - self.birth_time).total_seconds() / 3600
        return (
            f"Indi(age={age_hours:.2f}h, "
            f"episodes={self.memory.stats['total_episodes']}, "
            f"concepts={self.memory.stats['concepts_discovered']}, "
            f"words={self.memory.stats['words_learned']})"
        )
