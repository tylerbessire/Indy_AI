"""
Training Ground - The Interactive Learning Environment

Where Indi experiences the world through:
- Real-time speech (live mic)
- Picture uploads
- Picture-word associations
- Interactive teaching sessions
- Contrast learning ("same or different?")
- Curiosity-driven questions
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
from PIL import Image
import base64

from .fawn_brain import FAWNBrain
from .pure_memory import PureDevelopmentalMemory


class TeachingSession:
    """
    A one-on-one teaching session.
    Teacher shows, names, corrects, encourages.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.interactions = []
        self.items_taught = []

    def log_interaction(self, interaction_type: str, data: Dict):
        """Log a teaching interaction"""
        self.interactions.append({
            'timestamp': datetime.now().isoformat(),
            'type': interaction_type,
            'data': data
        })

    def add_taught_item(self, item: str, word: str):
        """Record what was taught"""
        self.items_taught.append({
            'item': item,
            'word': word,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self) -> Dict:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_interactions': len(self.interactions),
            'items_taught': len(self.items_taught),
            'taught_words': [item['word'] for item in self.items_taught]
        }


class TrainingGround:
    """
    The complete interactive learning environment.

    This is where Indi grows through multisensory teaching.
    """

    def __init__(
        self,
        brain: FAWNBrain,
        memory: PureDevelopmentalMemory,
        learning_rate: float = 0.0001
    ):
        self.brain = brain
        self.memory = memory
        self.optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)

        # Current teaching session
        self.current_session: Optional[TeachingSession] = None

        # Interaction history
        self.interaction_count = 0
        self.corrections_received = 0

        # Curiosity tracking
        self.curiosity_threshold = 0.7
        self.pending_questions = []

    def start_session(self) -> str:
        """Start a new teaching session"""
        session_id = f"session_{datetime.now().timestamp()}"
        self.current_session = TeachingSession(session_id)
        return session_id

    def end_session(self) -> Dict:
        """End current teaching session"""
        if self.current_session:
            summary = self.current_session.get_summary()
            self.current_session = None
            return summary
        return {'error': 'No active session'}

    def show_picture_with_word(
        self,
        image_data: Any,
        word: str,
        repetition: int = 1
    ) -> Dict:
        """
        Teacher shows a picture and says a word.
        Core grounding mechanism.

        Args:
            image_data: PIL Image or base64 string or numpy array
            word: The word teacher says
            repetition: How many times this has been shown (for reinforcement)

        Returns:
            Response with Indi's understanding
        """
        self.interaction_count += 1

        # Process image to tensor
        image_tensor = self._process_image(image_data)

        # Encode image to z-space
        with torch.no_grad():
            z_visual = self.brain.encode_visual(image_tensor)

        # Encode word to z-space (character-level)
        char_indices = self._text_to_indices(word)
        with torch.no_grad():
            z_text = self.brain.encode_text(char_indices)

        # Calculate curiosity about this
        curiosity = self.brain.assess_curiosity(z_visual).item()

        # Store experience
        episode_id = self.memory.store_experience(
            sensory_input={
                'type': 'picture_word_association',
                'word': word,
                'repetition': repetition
            },
            z_state=z_visual,
            teacher_words=word,
            salience=min(curiosity + (repetition * 0.1), 1.0)
        )

        # Contrastive learning: image and word should be close in z-space
        loss = torch.nn.functional.mse_loss(z_visual, z_text)

        # Micro-update
        self.brain.micro_update(loss, self.optimizer)

        # Log in session
        if self.current_session:
            self.current_session.log_interaction('picture_word', {
                'word': word,
                'repetition': repetition
            })
            self.current_session.add_taught_item('image', word)

        # Should Indi ask a question?
        question = None
        if curiosity > self.curiosity_threshold:
            question = self._generate_curiosity_question(word, curiosity)

        response = {
            'episode_id': episode_id,
            'understood': curiosity < 0.5,
            'curiosity': curiosity,
            'question': question,
            'learning_progress': self.brain.learning_steps,
            'concept_id': self.memory.semantic.find_nearest_concept(
                z_visual.detach().cpu().numpy()
            )[0]
        }

        return response

    def speak_to_indi(self, speech_text: str, audio_data: Optional[np.ndarray] = None) -> Dict:
        """
        Teacher speaks to Indi (text and/or audio).

        Args:
            speech_text: What was said (text)
            audio_data: Optional raw audio waveform

        Returns:
            Indi's response
        """
        self.interaction_count += 1

        # Encode text
        char_indices = self._text_to_indices(speech_text)
        with torch.no_grad():
            z_text = self.brain.encode_text(char_indices)

        # Encode audio if provided
        z_audio = None
        if audio_data is not None:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                z_audio = self.brain.encode_audio(audio_tensor)

        # Use audio if available, otherwise text
        z_speech = z_audio if z_audio is not None else z_text

        # Assess curiosity
        curiosity = self.brain.assess_curiosity(z_speech).item()

        # Try to generate a response
        response_text = self._generate_response(z_speech, speech_text)

        # Store experience
        episode_id = self.memory.store_experience(
            sensory_input={
                'type': 'speech',
                'text': speech_text,
                'has_audio': audio_data is not None
            },
            z_state=z_speech,
            teacher_words=speech_text,
            indi_output=response_text,
            salience=0.6
        )

        # Log in session
        if self.current_session:
            self.current_session.log_interaction('speech', {
                'text': speech_text,
                'response': response_text
            })

        return {
            'episode_id': episode_id,
            'response_text': response_text,
            'curiosity': curiosity,
            'understood': curiosity < 0.5
        }

    def correct_indi(self, correction: str, previous_episode_id: Optional[int] = None) -> Dict:
        """
        Teacher corrects Indi's understanding.

        Args:
            correction: The correction text
            previous_episode_id: Which episode to correct

        Returns:
            Learning outcome
        """
        self.corrections_received += 1

        # Encode correction
        char_indices = self._text_to_indices(correction)
        with torch.no_grad():
            z_correction = self.brain.encode_text(char_indices)

        # If we have previous episode, retrieve it and learn from difference
        if previous_episode_id is not None:
            prev_episode = self.memory.episodic.retrieve(previous_episode_id)
            if prev_episode:
                # Learn: previous understanding should move toward correction
                z_prev = torch.tensor(prev_episode.z_state, dtype=torch.float32).unsqueeze(0)
                loss = torch.nn.functional.mse_loss(z_prev, z_correction)

                # Strong update for corrections
                self.brain.micro_update(loss * 2.0, self.optimizer)

        # Store correction episode
        episode_id = self.memory.store_experience(
            sensory_input={'type': 'correction', 'text': correction},
            z_state=z_correction,
            teacher_words=correction,
            correction=correction,
            salience=0.9  # High salience for corrections
        )

        return {
            'episode_id': episode_id,
            'correction_count': self.corrections_received,
            'message': 'Learning from correction'
        }

    def ask_same_or_different(self, item1: Any, item2: Any) -> Dict:
        """
        Show two items and ask "same or different?"
        Core contrast learning.

        Args:
            item1, item2: Images, sounds, or text

        Returns:
            Indi's judgment + confidence
        """
        # Process both items
        z1 = self._encode_any_modality(item1)
        z2 = self._encode_any_modality(item2)

        # Compare
        with torch.no_grad():
            comparison_logits = self.brain.compare(z1, z2)
            prediction = torch.argmax(comparison_logits, dim=-1).item()
            confidence = torch.softmax(comparison_logits, dim=-1).max().item()

        # 0 = same, 1 = different
        judgment = "same" if prediction == 0 else "different"

        # Store this comparison
        episode_id = self.memory.store_experience(
            sensory_input={
                'type': 'comparison',
                'judgment': judgment,
                'confidence': confidence
            },
            z_state=z1,  # Store first item's representation
            indi_output=judgment,
            salience=0.5
        )

        return {
            'episode_id': episode_id,
            'judgment': judgment,
            'confidence': confidence,
            'uncertain': confidence < 0.6
        }

    def teach_contrast(
        self,
        item1: Any,
        item2: Any,
        are_same: bool,
        label1: Optional[str] = None,
        label2: Optional[str] = None
    ) -> Dict:
        """
        Explicitly teach a contrast with feedback.

        Args:
            item1, item2: The items to contrast
            are_same: Ground truth
            label1, label2: Optional labels for items

        Returns:
            Learning outcome
        """
        # Get Indi's judgment
        z1 = self._encode_any_modality(item1)
        z2 = self._encode_any_modality(item2)

        comparison_logits = self.brain.compare(z1, z2)

        # Target: 0 for same, 1 for different
        target = torch.tensor([[1.0, 0.0] if are_same else [0.0, 1.0]])

        # Learn from this
        loss = torch.nn.functional.cross_entropy(comparison_logits, target)
        self.brain.micro_update(loss, self.optimizer)

        # Store
        episode_id = self.memory.store_experience(
            sensory_input={
                'type': 'contrast_teaching',
                'are_same': are_same,
                'label1': label1,
                'label2': label2
            },
            z_state=z1,
            teacher_words=f"{label1} and {label2} are {'same' if are_same else 'different'}",
            salience=0.8
        )

        return {
            'episode_id': episode_id,
            'learned': True,
            'labels': [label1, label2] if label1 and label2 else []
        }

    def get_indi_state(self) -> Dict:
        """Get Indi's current developmental state"""
        memory_stats = self.memory.get_statistics()

        return {
            'brain': self.brain.get_state(),
            'memory': memory_stats,
            'self_concept': memory_stats.get('self_concept', {}),
            'interaction_count': self.interaction_count,
            'corrections_received': self.corrections_received,
            'current_session': self.current_session.get_summary() if self.current_session else None,
            'curiosity_threshold': self.curiosity_threshold
        }

    def _process_image(self, image_data: Any) -> torch.Tensor:
        """Convert various image formats to tensor [1, 3, H, W]"""
        if isinstance(image_data, torch.Tensor):
            return image_data

        if isinstance(image_data, str):
            # Base64 string
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data.astype('uint8'))
        else:
            image = image_data  # Assume PIL Image

        # Resize to standard size
        image = image.resize((224, 224))
        image = image.convert('RGB')

        # To tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        return image_tensor

    def _text_to_indices(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Convert text to character indices"""
        # Simple byte-level encoding
        indices = [ord(c) % 256 for c in text[:max_len]]

        # Pad if necessary
        while len(indices) < max_len:
            indices.append(0)

        return torch.tensor([indices], dtype=torch.long)

    def _encode_any_modality(self, item: Any) -> torch.Tensor:
        """Encode any type of input to z-space"""
        # Determine type and encode appropriately
        if isinstance(item, str):
            char_indices = self._text_to_indices(item)
            return self.brain.encode_text(char_indices)
        elif isinstance(item, (Image.Image, np.ndarray)):
            image_tensor = self._process_image(item)
            return self.brain.encode_visual(image_tensor)
        else:
            # Try to process as image
            image_tensor = self._process_image(item)
            return self.brain.encode_visual(image_tensor)

    def _generate_response(self, z_speech: torch.Tensor, input_text: str) -> str:
        """Generate a simple response based on z-space and memory"""
        # Try to find a word Indi knows for this concept
        word = self.memory.get_word_for_z(z_speech)

        if word:
            return f"{word}"
        elif self.interaction_count < 10:
            return "[listening]"
        else:
            # Try to recall similar episode
            similar = self.memory.recall_similar(z_speech, top_k=1)
            if similar:
                episode, score = similar[0]
                if score > 0.7 and episode.teacher_words:
                    return f"[recalls: {episode.teacher_words[:30]}...]"

            return "[thinking...]"

    def _generate_curiosity_question(self, context: str, curiosity_level: float) -> str:
        """Generate a question when curious"""
        if curiosity_level > 0.9:
            return "What?"
        elif curiosity_level > 0.7:
            return f"What is {context}?"
        else:
            return None
