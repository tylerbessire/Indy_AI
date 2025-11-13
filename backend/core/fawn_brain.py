"""
FAWN Brain - The Pure Neural Scaffolding
(Function-Agnostic Weighted Network)

This is Indi's neural substrate with ZERO pretrained knowledge.
All structure, no content. Random weights at initialization.
Everything learned through experience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class VisualCortex(nn.Module):
    """
    Minimal visual processor: pixels → latent space
    NO pretrained CNNs. Pure random initialization.
    """

    def __init__(self, z_dim: int = 64):
        super().__init__()

        # Tiny convolutional layers - learns from scratch
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Adaptive pooling to handle various image sizes
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Project to latent space
        self.to_z = nn.Linear(64 * 4 * 4, z_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixels: [B, C, H, W] raw pixel values (0-1 range)
        Returns:
            z: [B, z_dim] latent representation
        """
        # Expect input shape [B, 3, H, W]
        x = F.relu(self.conv1(pixels))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        z = self.to_z(x)
        return z


class AuditoryCortex(nn.Module):
    """
    Minimal audio processor: raw waveform → latent space
    NO pretrained models. Learns acoustic features from scratch.
    """

    def __init__(self, z_dim: int = 64):
        super().__init__()

        # Process raw waveform with 1D convolutions
        self.conv1 = nn.Conv1d(1, 32, kernel_size=25, stride=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, stride=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2)

        # Adaptive pooling for variable-length audio
        self.pool = nn.AdaptiveAvgPool1d(16)

        # Project to latent space
        self.to_z = nn.Linear(128 * 16, z_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T] raw audio samples
        Returns:
            z: [B, z_dim] latent representation
        """
        x = F.relu(self.conv1(waveform))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        z = self.to_z(x)
        return z


class TextStreamEncoder(nn.Module):
    """
    Character-level text processor: bytes/chars → latent space
    NO tokenizer. NO pretrained embeddings.
    Pure character-by-character learning.
    """

    def __init__(self, vocab_size: int = 256, z_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        # Simple character embedding (learned from scratch)
        self.char_embed = nn.Embedding(vocab_size, 32)

        # Small transformer for sequential processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Project to latent space
        self.to_z = nn.Linear(32, z_dim)

    def forward(self, char_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_indices: [B, seq_len] character/byte indices
        Returns:
            z: [B, z_dim] latent representation
        """
        # Embed characters
        x = self.char_embed(char_indices)

        # Process sequence
        x = self.transformer(x)

        # Pool over sequence (mean pooling)
        z = self.to_z(x.mean(dim=1))

        return z


class Comparator(nn.Module):
    """
    Compares two latent vectors: are they the same or different?
    Foundation for contrast learning.
    """

    def __init__(self, z_dim: int = 64):
        super().__init__()

        self.compare = nn.Sequential(
            nn.Linear(z_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [same, different]
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: [B, z_dim] latent vectors
        Returns:
            logits: [B, 2] same/different predictions
        """
        combined = torch.cat([z1, z2], dim=-1)
        return self.compare(combined)


class CuriosityHead(nn.Module):
    """
    Estimates uncertainty/novelty in latent space.
    High uncertainty → high curiosity → ask questions.
    """

    def __init__(self, z_dim: int = 64):
        super().__init__()

        self.uncertainty_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 uncertainty
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
        Returns:
            uncertainty: [B, 1] curiosity/uncertainty score
        """
        return self.uncertainty_net(z)


class ProtoLanguageDecoder(nn.Module):
    """
    Decodes latent space back to character sequences.
    Learns to generate language from scratch.
    """

    def __init__(self, z_dim: int = 64, vocab_size: int = 256, hidden_dim: int = 128):
        super().__init__()

        self.vocab_size = vocab_size

        # Project z to initial hidden state
        self.z_to_hidden = nn.Linear(z_dim, 32)

        # Small transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Character embedding for autoregressive generation
        self.char_embed = nn.Embedding(vocab_size, 32)

        # Output projection to vocabulary
        self.to_vocab = nn.Linear(32, vocab_size)

    def forward(self, z: torch.Tensor, char_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim] latent concept
            char_sequence: [B, seq_len] previous characters
        Returns:
            logits: [B, seq_len, vocab_size] next character predictions
        """
        # Create memory from z
        memory = self.z_to_hidden(z).unsqueeze(1)  # [B, 1, 32]

        # Embed character sequence
        tgt = self.char_embed(char_sequence)  # [B, seq_len, 32]

        # Decode
        output = self.transformer(tgt, memory)

        # Project to vocabulary
        logits = self.to_vocab(output)

        return logits


class FAWNBrain(nn.Module):
    """
    The complete FAWN (Function-Agnostic Weighted Network) Brain.

    Pure neural scaffolding with ZERO pretrained knowledge.
    Learns everything through sensory experience.
    """

    def __init__(self, z_dim: int = 64):
        super().__init__()

        self.z_dim = z_dim

        # Sensory cortices (all random init)
        self.visual_cortex = VisualCortex(z_dim=z_dim)
        self.auditory_cortex = AuditoryCortex(z_dim=z_dim)
        self.text_encoder = TextStreamEncoder(z_dim=z_dim)

        # Cognitive functions
        self.comparator = Comparator(z_dim=z_dim)
        self.curiosity_head = CuriosityHead(z_dim=z_dim)
        self.language_decoder = ProtoLanguageDecoder(z_dim=z_dim)

        # Plasticity tracking
        self.activation_history = []
        self.learning_steps = 0

    def encode_visual(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode raw pixels to latent space"""
        return self.visual_cortex(pixels)

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode raw audio to latent space"""
        return self.auditory_cortex(waveform)

    def encode_text(self, char_indices: torch.Tensor) -> torch.Tensor:
        """Encode character sequence to latent space"""
        return self.text_encoder(char_indices)

    def compare(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compare two latent representations"""
        return self.comparator(z1, z2)

    def assess_curiosity(self, z: torch.Tensor) -> torch.Tensor:
        """Assess curiosity/uncertainty about a concept"""
        return self.curiosity_head(z)

    def decode_to_language(self, z: torch.Tensor, char_sequence: torch.Tensor) -> torch.Tensor:
        """Decode latent concept to language"""
        return self.language_decoder(z, char_sequence)

    def micro_update(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Micro-SGD plasticity update.
        Small gradient step = continuous brain growth.
        """
        optimizer.zero_grad()
        loss.backward()

        # Clip for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()

        self.learning_steps += 1

    def get_state(self) -> Dict:
        """Get current brain state for visualization"""
        return {
            'z_dim': self.z_dim,
            'learning_steps': self.learning_steps,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
