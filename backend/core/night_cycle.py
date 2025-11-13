"""
Night Cycle - Sleep and Dreams

Where Indi consolidates learning during sleep:
- NREM Stage: Replay, stabilization, compression
- REM Stage: Dreams, recombination, analogies
- Schema Consolidation: Update concepts and prototypes

CRITICAL: Dreams only recombine what Indi has experienced.
NO artificial knowledge injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from .fawn_brain import FAWNBrain
from .pure_memory import PureDevelopmentalMemory, Episode


class SleepCycle:
    """
    Manages Indi's sleep-wake cycle.
    Sleep is when learning consolidates.
    """

    def __init__(
        self,
        brain: FAWNBrain,
        memory: PureDevelopmentalMemory,
        nrem_duration_minutes: int = 20,
        rem_duration_minutes: int = 10
    ):
        self.brain = brain
        self.memory = memory
        self.nrem_duration = timedelta(minutes=nrem_duration_minutes)
        self.rem_duration = timedelta(minutes=rem_duration_minutes)

        # Sleep tracking
        self.sleep_count = 0
        self.last_sleep = None
        self.sleep_history = []

    def initiate_sleep(self) -> Dict:
        """
        Begin a complete sleep cycle.

        Returns:
            Summary of sleep and consolidation
        """
        self.sleep_count += 1
        sleep_start = datetime.now()

        print(f"[Sleep Cycle {self.sleep_count}] Indi is falling asleep...")

        # Stage 1: NREM - Replay and Stabilization
        nrem_results = self._nrem_stage()

        # Stage 2: REM - Dreams and Recombination
        rem_results = self._rem_stage()

        # Stage 3: Schema Consolidation
        consolidation_results = self._consolidate_schemas()

        sleep_end = datetime.now()
        sleep_duration = (sleep_end - sleep_start).total_seconds()

        summary = {
            'sleep_number': self.sleep_count,
            'start_time': sleep_start.isoformat(),
            'end_time': sleep_end.isoformat(),
            'duration_seconds': sleep_duration,
            'nrem': nrem_results,
            'rem': rem_results,
            'consolidation': consolidation_results
        }

        self.last_sleep = sleep_start
        self.sleep_history.append(summary)

        print(f"[Sleep Cycle {self.sleep_count}] Indi wakes up refreshed!")

        return summary

    def _nrem_stage(self) -> Dict:
        """
        NREM (Non-REM) Sleep Stage:
        - Replay recent experiences
        - Stabilize new patterns
        - Reduce noise
        - Strengthen important memories
        """
        print("  [NREM] Replaying experiences...")

        # Get recent salient episodes
        recent_episodes = self.memory.episodic.retrieve_by_salience(top_k=50)

        if len(recent_episodes) < 5:
            return {
                'replayed_count': 0,
                'message': 'Not enough experiences to replay'
            }

        # Replay and strengthen
        replay_count = 0
        total_loss = 0.0

        optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.0001)

        for episode in recent_episodes[:20]:  # Replay top 20
            try:
                # Retrieve z-state
                z = torch.tensor(episode.z_state, dtype=torch.float32).unsqueeze(0)

                # Autoencoding loss (reconstruct own representation)
                # This strengthens the pattern
                z_reconstructed = self._autoencode_z(z)

                loss = F.mse_loss(z, z_reconstructed)

                # Predictive modeling (if we have sequences)
                # Try to predict next state from current state
                # (simplified version)

                # Update
                self.brain.micro_update(loss * episode.salience, optimizer)

                total_loss += loss.item()
                replay_count += 1

            except Exception as e:
                print(f"  [NREM] Error replaying episode: {e}")
                continue

        avg_loss = total_loss / max(replay_count, 1)

        return {
            'replayed_count': replay_count,
            'average_reconstruction_loss': avg_loss,
            'message': f'Replayed {replay_count} experiences'
        }

    def _rem_stage(self) -> Dict:
        """
        REM Sleep Stage:
        - Generate dreams (novel combinations)
        - Form analogies
        - Recombine memories
        - Explore latent space

        CRITICAL: Only recombine existing experiences!
        """
        print("  [REM] Dreaming...")

        # Get recent episodes
        recent_episodes = self.memory.episodic.get_recent(30)

        if len(recent_episodes) < 2:
            return {
                'dreams_generated': 0,
                'message': 'Not enough experiences to dream about'
            }

        dreams_generated = 0
        dream_log = []

        # Generate dreams by interpolating between experiences
        for i in range(min(10, len(recent_episodes) - 1)):
            try:
                # Pick two random episodes
                idx1 = np.random.randint(0, len(recent_episodes))
                idx2 = np.random.randint(0, len(recent_episodes))

                if idx1 == idx2:
                    continue

                ep1 = recent_episodes[idx1]
                ep2 = recent_episodes[idx2]

                # Interpolate in z-space (creating novel combination)
                z1 = torch.tensor(ep1.z_state, dtype=torch.float32)
                z2 = torch.tensor(ep2.z_state, dtype=torch.float32)

                # Random interpolation
                alpha = np.random.beta(2, 2)  # Bias toward center
                z_dream = alpha * z1 + (1 - alpha) * z2

                # Add small noise for exploration
                z_dream = z_dream + torch.randn_like(z_dream) * 0.1

                # "Experience" this dream (update memory clustering)
                self.memory.semantic.update_clusters(z_dream.numpy().reshape(1, -1))

                dreams_generated += 1

                # Log dream
                word1 = ep1.teacher_words or "unknown"
                word2 = ep2.teacher_words or "unknown"
                dream_log.append({
                    'dream_id': dreams_generated,
                    'combined': f"{word1} + {word2}",
                    'alpha': float(alpha)
                })

            except Exception as e:
                print(f"  [REM] Error generating dream: {e}")
                continue

        return {
            'dreams_generated': dreams_generated,
            'dream_samples': dream_log[:5],  # First 5 dreams
            'message': f'Generated {dreams_generated} dreams'
        }

    def _consolidate_schemas(self) -> Dict:
        """
        Schema Consolidation:
        - Update concept clusters
        - Strengthen concept-word associations
        - Prune weak concepts
        - Update prototypes
        """
        print("  [Consolidation] Updating schemas...")

        # Get all recent z-states
        recent_episodes = self.memory.episodic.get_recent(100)

        if len(recent_episodes) < self.memory.semantic.n_clusters:
            return {
                'message': 'Not enough data for consolidation',
                'concepts_updated': 0
            }

        # Extract z-states
        z_states = np.array([ep.z_state for ep in recent_episodes])

        # Update clusters
        initial_concepts = len(self.memory.semantic.concepts)
        self.memory.semantic.update_clusters(z_states)
        final_concepts = len(self.memory.semantic.concepts)

        # Update concept-episode associations
        for episode_idx, episode in enumerate(recent_episodes):
            concept_id, distance = self.memory.semantic.find_nearest_concept(episode.z_state)

            if concept_id >= 0 and episode.teacher_words:
                # Re-associate
                word = episode.teacher_words.strip().split()[0] if episode.teacher_words.strip() else None
                if word:
                    self.memory.semantic.associate_word(
                        concept_id,
                        self.memory.episodic.episodes.index(episode),
                        word.lower()
                    )

        return {
            'concepts_before': initial_concepts,
            'concepts_after': final_concepts,
            'concepts_updated': final_concepts,
            'episodes_processed': len(recent_episodes),
            'message': f'Updated {final_concepts} concepts from {len(recent_episodes)} experiences'
        }

    def _autoencode_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Simple autoencoding through the brain.
        Forces compression and reconstruction.
        """
        # Use the brain's comparator as an autoencoder pathway
        # (In a full implementation, you'd have a dedicated autoencoder)

        # For now, just add noise and denoise
        z_noisy = z + torch.randn_like(z) * 0.1

        # "Reconstruct" by finding nearest concept and using its prototype
        concept_id, distance = self.memory.semantic.find_nearest_concept(
            z_noisy.detach().cpu().numpy()[0]
        )

        if concept_id >= 0 and concept_id in self.memory.semantic.concepts:
            prototype = self.memory.semantic.concepts[concept_id].prototype
            z_reconstructed = torch.tensor(prototype, dtype=torch.float32).unsqueeze(0)
        else:
            z_reconstructed = z_noisy

        return z_reconstructed


class DayNightScheduler:
    """
    Manages Indi's circadian rhythm:
    - Day: Learning, interaction, experience
    - Night: Sleep, dreams, consolidation
    """

    def __init__(
        self,
        day_duration_hours: float = 4.0,
        night_duration_hours: float = 1.0
    ):
        self.day_duration = timedelta(hours=day_duration_hours)
        self.night_duration = timedelta(hours=night_duration_hours)

        self.current_phase = 'day'
        self.phase_start_time = datetime.now()
        self.cycle_count = 0

        self.history = []

    def check_phase_transition(self) -> Tuple[bool, str]:
        """
        Check if it's time to transition between day and night.

        Returns:
            (should_transition, new_phase)
        """
        elapsed = datetime.now() - self.phase_start_time

        if self.current_phase == 'day' and elapsed >= self.day_duration:
            return True, 'night'
        elif self.current_phase == 'night' and elapsed >= self.night_duration:
            return True, 'day'

        return False, self.current_phase

    def transition_to(self, new_phase: str) -> Dict:
        """Transition to a new phase"""
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_start_time = datetime.now()

        if new_phase == 'day':
            self.cycle_count += 1

        transition = {
            'cycle': self.cycle_count,
            'from': old_phase,
            'to': new_phase,
            'timestamp': datetime.now().isoformat()
        }

        self.history.append(transition)

        return transition

    def get_current_phase_info(self) -> Dict:
        """Get information about current phase"""
        elapsed = datetime.now() - self.phase_start_time

        if self.current_phase == 'day':
            total = self.day_duration
        else:
            total = self.night_duration

        progress = min(elapsed.total_seconds() / total.total_seconds(), 1.0)

        return {
            'phase': self.current_phase,
            'cycle': self.cycle_count,
            'elapsed_seconds': elapsed.total_seconds(),
            'progress': progress,
            'phase_start': self.phase_start_time.isoformat()
        }
