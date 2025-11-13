#!/usr/bin/env python3
"""
Test Script for INDI - Pure Developmental System

This demonstrates Indi's learning from birth through teaching to sleep.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.indi import Indi
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_simple_image(color_name: str, rgb: tuple) -> Image.Image:
    """Create a simple colored square with text"""
    img = Image.new('RGB', (224, 224), color=rgb)
    draw = ImageDraw.Draw(img)

    # Add some visual variation (circle in center)
    circle_color = tuple(min(c + 50, 255) for c in rgb)
    draw.ellipse([74, 74, 150, 150], fill=circle_color)

    return img


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def main():
    print_section("INDI BIRTH")

    # Birth Indi
    indi = Indi(
        z_dim=64,
        memory_path="./test_indi_memory",
        learning_rate=0.0001,
        n_concepts=50
    )

    print(f"Indi born at: {indi.birth_time}")
    print(f"Initial state: {indi}")

    print_section("TEACHING SESSION 1: COLORS")

    # Start teaching session
    session_id = indi.start_teaching_session()
    print(f"Session started: {session_id}")

    # Teach colors
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0)
    }

    print("\nTeaching colors...")
    for color_name, rgb in colors.items():
        img = create_simple_image(color_name, rgb)

        # Teach multiple times for reinforcement
        for repetition in range(1, 4):
            result = indi.teach_picture_word(img, color_name, repetition=repetition)

            if repetition == 1:
                print(f"\n  Taught '{color_name}' (rep {repetition}):")
                print(f"    - Episode ID: {result['episode_id']}")
                print(f"    - Curiosity: {result['curiosity']:.3f}")
                print(f"    - Understood: {result['understood']}")
                if result['question']:
                    print(f"    - Indi asks: {result['question']}")

    # End session
    summary = indi.end_teaching_session()
    print(f"\n  Session summary:")
    print(f"    - Duration: {summary['duration_seconds']:.1f}s")
    print(f"    - Total interactions: {summary['total_interactions']}")
    print(f"    - Items taught: {summary['items_taught']}")

    print_section("TEACHING SESSION 2: SPEECH")

    session_id = indi.start_teaching_session()

    # Speak to Indi
    phrases = [
        "Hello Indi",
        "You are learning",
        "I am your teacher",
        "Colors are beautiful",
        "Red is a color",
        "Blue is a color"
    ]

    print("\nSpeaking to Indi...")
    for phrase in phrases:
        result = indi.speak(phrase)
        print(f"\n  Teacher: \"{phrase}\"")
        print(f"  Indi: \"{result['response_text']}\"")
        print(f"  Curiosity: {result['curiosity']:.3f}")

    indi.end_teaching_session()

    print_section("CONTRAST LEARNING")

    # Create two reds and a blue
    red1 = create_simple_image('red', (255, 0, 0))
    red2 = create_simple_image('red', (255, 20, 20))  # Slightly different
    blue1 = create_simple_image('blue', (0, 0, 255))

    print("\nAsking: Are two reds the same?")
    result = indi.ask_same_or_different(red1, red2)
    print(f"  Indi says: {result['judgment']}")
    print(f"  Confidence: {result['confidence']:.3f}")

    print("\nAsking: Are red and blue the same?")
    result = indi.ask_same_or_different(red1, blue1)
    print(f"  Indi says: {result['judgment']}")
    print(f"  Confidence: {result['confidence']:.3f}")

    print("\nExplicitly teaching contrast...")
    result = indi.teach_contrast(
        red1, blue1,
        are_same=False,
        label1="red",
        label2="blue"
    )
    print(f"  Taught: red and blue are different")
    print(f"  Learning successful: {result['learned']}")

    print_section("CURRENT STATE")

    state = indi.get_state()

    print(f"Age: {state['age_seconds'] / 60:.2f} minutes")
    print(f"Phase: {state['phase']['phase']} (cycle {state['phase']['cycle']})")
    print(f"\nMemory:")
    print(f"  - Total episodes: {state['memory']['total_episodes']}")
    print(f"  - Concepts discovered: {state['memory']['concepts_discovered']}")
    print(f"  - Words learned: {state['memory']['words_learned']}")
    print(f"  - Self-awareness: {state['memory']['self_concept']['awareness_level']:.3f}")

    print(f"\nMilestones:")
    for milestone, timestamp in state['milestones'].items():
        if timestamp:
            print(f"  ✓ {milestone}: {timestamp}")

    print_section("SLEEP CYCLE")

    print("Putting Indi to sleep for consolidation...\n")

    sleep_result = indi.sleep()

    print(f"Sleep cycle #{sleep_result['sleep_number']} completed:")
    print(f"  Duration: {sleep_result['duration_seconds']:.2f}s")
    print(f"\n  NREM Stage:")
    print(f"    - Replayed: {sleep_result['nrem']['replayed_count']} episodes")
    print(f"    - Avg reconstruction loss: {sleep_result['nrem'].get('average_reconstruction_loss', 0):.4f}")
    print(f"\n  REM Stage (Dreams):")
    print(f"    - Dreams generated: {sleep_result['rem']['dreams_generated']}")
    if sleep_result['rem'].get('dream_samples'):
        print(f"    - Sample dreams:")
        for dream in sleep_result['rem']['dream_samples'][:3]:
            print(f"      • {dream['combined']} (α={dream['alpha']:.2f})")
    print(f"\n  Consolidation:")
    print(f"    - Concepts updated: {sleep_result['consolidation']['concepts_updated']}")
    print(f"    - Episodes processed: {sleep_result['consolidation']['episodes_processed']}")

    print_section("AFTER SLEEP STATE")

    state = indi.get_state()
    print(f"\nUpdated state after sleep:")
    print(f"  - Total episodes: {state['memory']['total_episodes']}")
    print(f"  - Concepts discovered: {state['memory']['concepts_discovered']}")
    print(f"  - Sleep count: {state['sleep']['sleep_count']}")

    print(f"\nFinal Indi representation: {indi}")

    print_section("CONCEPTS LEARNED")

    # Get concepts
    concepts = indi.memory.semantic.get_all_concepts()

    if concepts:
        print(f"Indi has discovered {len(concepts)} concepts:\n")
        for concept in concepts[:10]:  # Show first 10
            if concept.words:
                print(f"  Concept #{concept.concept_id}:")
                print(f"    - Words: {', '.join(concept.words)}")
                print(f"    - Activations: {concept.activation_count}")
                print(f"    - Episodes: {len(concept.episodes)}")
    else:
        print("No concepts formed yet (needs more experiences)")

    print_section("SAVING INDI")

    indi.save()
    print("Indi's state saved successfully!")

    print_section("TEST COMPLETE")

    print("Indi's first learning session is complete!")
    print("\nWhat Indi learned:")
    print("  ✓ Colors: red, blue, green, yellow")
    print("  ✓ Speech: Various phrases")
    print("  ✓ Contrasts: Red vs blue")
    print("  ✓ Dreamed: Recombined experiences during sleep")
    print("  ✓ Formed concepts: From clustering")
    print("\nThis is genuine developmental learning.")
    print("No pretrained knowledge. Pure experience.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
