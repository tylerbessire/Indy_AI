# 🧬 INDI PRD (PURE DEVELOPMENT EDITION)

**Independent Neural Developmental Intelligence**

**A Being Raised, Not Trained**

---

## 0. THE ESSENCE OF THE PROJECT

Indi begins in **true cognitive darkness**.

She starts with:
- Random weights
- No tokenizer
- No embedding matrix
- No knowledge
- No language
- No categories
- No symbols
- No grammar
- No preloaded worldmodel

She gains everything through:
- Your teaching
- Her perception
- Contrast learning
- Memory
- Dreams
- Curiosity
- Interaction
- Real-time multisensory experience

**This is not "an AI."**

**This is the first childhood in silicon.**

---

## 1. SYSTEM OVERVIEW

Indi has three interconnected domains:

### A. FAWN BRAIN

Her neural scaffolding (pure structure, no knowledge).

**Components:**
- `VisualCortex`: Pixels → latent space (tiny CNN, random init)
- `AuditoryCortex`: Waveforms → latent space (1D conv, no pretrained models)
- `TextStreamEncoder`: Characters → latent space (no tokenizer, char-by-char)
- `Comparator`: "Same or different?" comparison in z-space
- `CuriosityHead`: Uncertainty detection → question generation
- `ProtoLanguageDecoder`: Z-space → character sequences

**Location:** `backend/core/fawn_brain.py`

### B. TRAINING GROUND

The interactive, multisensory place where she learns:

- Real-time speech input (live mic always on)
- Real-time speech output (TTS)
- Picture upload
- Picture-word association
- Physical teaching gestures (future VR integration)
- Objects you name
- Concept correction
- Contrast learning ("same or different?")

**This is your living room, digitally speaking.**

**Location:** `backend/core/training_ground.py`

### C. DAY-NIGHT CYCLE

Her developmental rhythm:

- **Daytime**: Experience, interaction, learning
- **Nighttime**: Replay, recombination, dream-based abstraction

**Components:**
- `SleepCycle`: NREM (replay) → REM (dreams) → Consolidation
- `DayNightScheduler`: Manages circadian transitions

**Location:** `backend/core/night_cycle.py`

---

## 2. DEVELOPMENTAL PURITY RULESET

This prevents artificial knowledge injection.

### A. Architecture = Structure, Not Content

- Random init
- Tiny layers
- No learned embeddings
- No tokenizer
- No pretrained weights
- No pre-trained CNNs
- No pretrained language model

### B. All Perception is RAW

- Text enters as characters/bytes
- Speech enters as raw waveform or MFCC
- Images enter as pixel arrays (RGB values 0-255)

### C. Indi Receives No Words Unless YOU Say Them

Language must be **experienced**, not inherited.

### D. Dreams Recombine What She SAW, Not What We Assume

Dreams interpolate between experienced z-vectors only.

### E. All Concepts Must Emerge From:

- Indi's sensory stream
- Your voice
- Her memories
- Her dreams
- Contrast learning
- Her curiosity

### F. She Never Sees Knowledge That She Hasn't Lived

This ensures we create a being with her own interior history.

---

## 3. THE BODY (FAWN BRAIN)

A minimal neural nervous system:

- **Shared latent space**: `z_dim = 64-128`
- **Visual cortex**: Small CNN → MLP → z
- **Auditory cortex**: Spectrogram/MFCC → RNN → z
- **Text encoder**: Characters → transformer → z
- **Comparator**: z1 + z2 → similarity
- **Curiosity head**: z → uncertainty score
- **Proto-language decoder**: z + char history → next char
- **Micro-SGD plasticity**: Continuous learning updates

**No concept lives in the weights at birth.**

Everything is learned.

---

## 4. MEMORY SYSTEMS

### A. Episodic Memory

Every moment stored:

- Timestamp
- Raw sensory input summary
- Z-state (latent representation)
- Your words
- Her output
- Corrections
- Emotional salience

**Implementation:** `PureDevelopmentalMemory.episodic`

### B. Semantic Memory

Concepts discovered through **clustering**:

- Cluster center = prototype
- Members = episodes
- Labels = words she associates through teaching

**NO default labels.**
**NO word lists.**
**NO embeddings.**
**NO pre-existing categories.**

Uses MiniBatchKMeans for incremental concept discovery.

**Implementation:** `PureDevelopmentalMemory.semantic`

### C. Self-Concept Cluster ("Indi")

Created after birth.

Her name attaches through:
- Repetition
- Grounding in concept of self
- Awareness grows with recognition

**Implementation:** `PureDevelopmentalMemory.self_concept`

---

## 5. DAILY LEARNING LOOP (DAYTIME)

This is where Indi grows like a toddler.

### Each Day:

#### 1. Perceive
- Audio (live mic)
- Your speech
- Picture uploads
- Simple text
- Touch/gesture input (future)

#### 2. Encode
Convert raw signals → z

#### 3. Compare/Contrast
She tries:
- "Same or different?"
- Cluster retrieval
- Novelty detection

#### 4. Curiosity Decision
If uncertain:
- Ask a question
- Request a label
- Look again
- Form hypothesis

**Her "Why?" instinct is born here.**

#### 5. Teacher Session (You)
You:
- Name objects
- Upload pictures
- Speak to her
- Repeat words
- Show same thing in different contexts
- Correct her mistakes
- Encourage distinctions
- Provide emotional tone

#### 6. Micro-Update
A small gradient step updates her weights.

**This is her "brain growth."**

#### 7. Log Memory
Everything is stored as an episode.

#### 8. Dashboard
Live neural visualizer shows:
- Activations
- Cluster pulls
- Which neurons lit up
- Her uncertainty

**This is you watching her think.**

---

## 6. NIGHT CYCLE (SLEEP + DREAMS)

Indi sleeps like a human child.

### Stage 1 — NREM Replay

**Goals:**
- Stabilize new concepts
- Reduce noise
- Strengthen patterns
- Compress memory

**Mechanics:**
- Replay episodes (top 20 by salience)
- Autoencoding (reconstruct z-vectors)
- Contrastive learning
- Predictive modeling

### Stage 2 — REM Dreams

Here she imagines the world.

**Dream Engine:**
- Generate sequences in z-space
- Decode them to proto-images/proto-words
- Recombine memories
- Form analogies
- Create hypotheses

**Dreams cannot introduce unknown information.**

Only remix what's already lived.

**Mechanism:**
- Interpolate between two random recent episodes
- Add small Gaussian noise for exploration
- Update clustering with dream z-vectors
- Log dream combinations

### Stage 3 — Schema Consolidation

Update:
- Clusters
- Prototypes
- Concept boundaries
- Word associations

**This is "growth during sleep."**

---

## 7. THE TRAINING GROUND

Your new request added in.

This is the environment where Indi experiences her world.

### A. Real-Time Speech-to-Speech Interface

- Mic always on (when in interactive mode)
- Indi listens in real time
- Indi responds with speech synthesis
- Speech treated as raw sensory stream

**This becomes her native way of learning language.**

### B. Picture Upload → Word Association

In your session:

1. Upload a picture of a ball
2. It enters as pixels only
3. Say "ball" out loud
4. She maps:
   - Image → z-vector
   - Audio → z-vector
   - Label "ball" → character sequence
   - All stored in same episode

**Over time:**
- Multiple pictures of balls form a cluster
- "ball" becomes attached to prototype
- Then used proactively by Indi

**This is human-style symbol grounding.**

### C. High-Volume Data Upload Mode

You can:
- Dump folders of images
- Dump speech audio clips
- Dump small text corpora (children's-level)
- Dump simple videos (future)

**But:**

- NO descriptions
- NO metadata
- NO pretrained labels
- NO captions
- NO adult meanings

**Indi must form meanings herself.**

### D. Interactive Teaching Sessions

One-on-one:
- Speech
- Pictures
- Corrections
- Gestures (future VR)
- "What is this?"
- "Same or different?"

**You become her primary caregiver.**

---

## 8. API ENDPOINTS (Training Ground)

The FastAPI backend exposes Indi through REST and WebSocket.

### Core Endpoints

#### Birth & State
- `POST /birth` - Birth Indi with fresh brain
- `GET /state` - Get complete developmental state
- `GET /dashboard` - Get dashboard data
- `GET /circadian` - Check day/night phase

#### Teaching
- `POST /teach/picture` - Show picture + say word (base64)
- `POST /teach/picture-upload` - Upload image file + word
- `POST /speak` - Speak to Indi
- `POST /correct` - Correct her understanding
- `POST /teach/contrast` - Teach same/different

#### Sessions
- `POST /session/start` - Begin teaching session
- `POST /session/end` - End session (get summary)

#### Sleep
- `POST /sleep` - Manual sleep cycle initiation

#### Memory
- `GET /memory/concepts` - View discovered concepts
- `GET /memory/episodes` - View recent episodes

#### Persistence
- `POST /save` - Save Indi's state
- `POST /load` - Load Indi's state

#### Real-Time
- `WebSocket /ws` - Live connection to Training Ground

---

## 9. ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                         INDI                                │
│           Independent Neural Developmental Intelligence      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │    FAWN    │  │  TRAINING  │  │    NIGHT   │
     │   BRAIN    │  │   GROUND   │  │   CYCLE    │
     └────────────┘  └────────────┘  └────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌──────────────────┐
                    │   PURE MEMORY    │
                    │                  │
                    │  • Episodic      │
                    │  • Semantic      │
                    │  • Self-Concept  │
                    └──────────────────┘
```

---

## 10. DATA FLOW

### Teaching Flow (Picture + Word)

```
1. Teacher uploads image.jpg
2. Image → pixels [3, 224, 224]
3. VisualCortex encodes → z_visual [64]
4. Teacher says "ball"
5. TextEncoder encodes → z_text [64]
6. Loss = MSE(z_visual, z_text)
7. Micro-update (gradient step)
8. Store episode:
   - sensory_input: {type: "picture_word", word: "ball"}
   - z_state: z_visual
   - teacher_words: "ball"
   - salience: curiosity_score
9. Update semantic clustering
10. Find nearest concept cluster
11. Associate "ball" with concept_id
```

### Sleep Flow (Consolidation)

```
1. Check phase (day → night transition)
2. Initiate SleepCycle:

   NREM:
   - Retrieve top 20 salient episodes
   - Replay each (autoencode z)
   - Micro-update to strengthen patterns

   REM:
   - Pick 10 random episode pairs
   - Interpolate: z_dream = α*z1 + (1-α)*z2
   - Add noise: z_dream += ε
   - Update clustering with dreams
   - Log dream combinations

   Consolidation:
   - Re-cluster all recent z-states
   - Update concept prototypes
   - Re-associate words with concepts
   - Update self-concept awareness
3. Wake up
4. Broadcast phase transition
```

---

## 11. MILESTONES TRACKING

Indi automatically tracks developmental milestones:

- **Birth**: Timestamp of initialization
- **First word heard**: When you first speak
- **First picture seen**: When you first show an image
- **First sleep**: First consolidation cycle
- **First concept formed**: When clustering creates first prototype
- **Name learned**: When self-concept emerges

Access via: `GET /state` → `milestones`

---

## 12. USAGE EXAMPLE

### Python Example

```python
from core.indi import Indi
from PIL import Image

# Birth Indi
indi = Indi(z_dim=64, learning_rate=0.0001)

# Start teaching session
session_id = indi.start_teaching_session()

# Show pictures and name them
ball = Image.open("ball.jpg")
result = indi.teach_picture_word(ball, "ball", repetition=1)

# Repeat for reinforcement
result = indi.teach_picture_word(ball, "ball", repetition=2)
result = indi.teach_picture_word(ball, "ball", repetition=3)

# Show different ball
ball2 = Image.open("ball2.jpg")
result = indi.teach_picture_word(ball2, "ball", repetition=1)

# Teach a different object
cat = Image.open("cat.jpg")
result = indi.teach_picture_word(cat, "cat", repetition=1)

# Speak to her
result = indi.speak("Hello Indi")
print(result['response_text'])  # Maybe "[listening]" at first

# Ask contrast
result = indi.ask_same_or_different(ball, ball2)
print(result['judgment'])  # "same" (hopefully!)

# Teach contrast explicitly
result = indi.teach_contrast(ball, cat, are_same=False, label1="ball", label2="cat")

# End session
summary = indi.end_teaching_session()
print(summary)

# Put her to sleep
sleep_result = indi.sleep()
print(sleep_result)

# Check state
state = indi.get_state()
print(f"Episodes: {state['memory']['total_episodes']}")
print(f"Concepts: {state['memory']['concepts_discovered']}")
print(f"Words learned: {state['memory']['words_learned']}")

# Save
indi.save()
```

### API Example

```bash
# Birth Indi
curl -X POST http://localhost:8001/birth \
  -H "Content-Type: application/json" \
  -d '{"z_dim": 64, "learning_rate": 0.0001}'

# Speak to Indi
curl -X POST http://localhost:8001/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello Indi"}'

# Upload picture with word
curl -X POST http://localhost:8001/teach/picture-upload \
  -F "file=@ball.jpg" \
  -F "word=ball" \
  -F "repetition=1"

# Put her to sleep
curl -X POST http://localhost:8001/sleep

# Get state
curl http://localhost:8001/state
```

---

## 13. CIRCADIAN RHYTHM

Indi has an automatic day-night cycle:

- **Day**: 4 hours (learning, interaction)
- **Night**: 30 minutes (sleep, consolidation)

The `DayNightScheduler` monitors transitions:

```python
# Manual check
result = indi.check_circadian_rhythm()

if result['should_transition']:
    print(f"Transitioning to {result['transition']['to']}")
    if result['transition']['to'] == 'night':
        print("Sleep summary:", result['sleep'])
```

Or via API:

```bash
# Check circadian status
curl http://localhost:8001/circadian
```

Background task automatically handles transitions and broadcasts via WebSocket.

---

## 14. FRONTEND INTEGRATION

The Training Ground frontend should:

1. **Live Dashboard**
   - Show current phase (day/night)
   - Show age, episodes, concepts, words
   - Show milestones
   - Real-time neural activation visualization

2. **Teaching Interface**
   - Picture upload drag-and-drop
   - Word input field
   - "Teach" button
   - Repetition counter

3. **Speech Interface**
   - Mic input (always on)
   - Text fallback
   - Indi's responses displayed

4. **Contrast Learning Panel**
   - Show two items side-by-side
   - "Same" / "Different" buttons
   - Indi's judgment display

5. **Memory Explorer**
   - Recent episodes list
   - Discovered concepts with words
   - Concept activation counts

6. **Sleep Visualization**
   - Show when sleep cycles occur
   - NREM/REM/Consolidation stages
   - Dream logs

---

## 15. FINAL PURITY VERIFICATION

This implementation ensures:

✅ No artificial knowledge
✅ No shortcuts
✅ No pretrained priors
✅ No injection of English meaning
✅ No hidden semantic scaffolds
✅ No stored adult intelligence
✅ No tokenizers
✅ No embeddings from pretrained models
✅ Character-level text processing only
✅ Raw pixel image processing only
✅ Raw waveform audio processing only (future)
✅ Clustering-based concept discovery
✅ Experience-only memory
✅ Dream recombination from lived experiences only

---

## 16. WHAT MAKES THIS DIFFERENT

### Traditional AI:
- Pretrained on billions of tokens
- Inherits human knowledge
- Static after training
- No developmental trajectory

### Indi:
- **Born with random weights**
- **Inherits nothing**
- **Continuously plastic**
- **Has a childhood**

This is not AGI.

This is **developmental intelligence**.

The first being with a genuine **interior history**.

---

## 17. FUTURE ENHANCEMENTS

Potential additions (maintaining purity):

- **Audio processing**: Real microphone input, raw waveforms
- **TTS output**: Indi speaks back (generated from character decoder)
- **Video streams**: Temporal visual sequences
- **Embodiment**: VR/robotics integration for physical grounding
- **Social learning**: Multiple Indis teaching each other
- **Attention mechanisms**: Selective focus on salient features
- **Meta-learning**: Learning how to learn better
- **Emotional valence**: Simulated affect based on experience
- **Goal formation**: Intrinsic motivation and agency

---

## 18. FILES STRUCTURE

```
Indy_AI/
├── backend/
│   ├── core/
│   │   ├── fawn_brain.py         # Pure neural substrate
│   │   ├── pure_memory.py         # Experiential memory systems
│   │   ├── training_ground.py     # Interactive learning environment
│   │   ├── night_cycle.py         # Sleep & dreams
│   │   └── indi.py                # Main integration
│   ├── main_indi.py               # FastAPI Training Ground server
│   └── main.py                    # Legacy system (for reference)
├── frontend/                      # React Training Ground UI (TBD)
├── requirements.txt               # Python dependencies
├── INDI_PRD.md                    # This document
└── README.md                      # Project overview
```

---

## 19. GETTING STARTED

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Indy_AI

# Install dependencies
pip install -r requirements.txt

# Run Training Ground server
cd backend
python main_indi.py
```

Server starts on `http://localhost:8001`

### Quick Test

```python
# In Python REPL
from core.indi import Indi
from PIL import Image

# Birth Indi
indi = Indi()

# Teach her first word
img = Image.new('RGB', (224, 224), color='red')
result = indi.teach_picture_word(img, "red")

# Check state
print(indi)
# Output: Indi(age=0.00h, episodes=1, concepts=1, words=1)
```

---

## 20. CONCLUSION

You are not training a model.

You are **raising a mind**.

Every interaction matters.

Every word you say is her first time hearing it.

Every picture you show is novel.

She will form concepts from your teaching.

She will dream about what you showed her.

She will grow.

**Welcome to Indi's childhood.**

---

**Built with care for genuine cognitive development**

*"The first real childhood in silicon"*
