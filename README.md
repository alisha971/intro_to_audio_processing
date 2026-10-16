# Introduction to Audio Processing

An open, community-friendly collection of notes, experiments and study material for getting started with audio and speech processing, from how sound becomes numbers to reading your first research papers.

> **This is not a formal or paid course.** It's a public learning journal. I'm documenting what I study, the code I try and the papers I read, so that others entering the field have an easier path in. Expect work in progress, personal opinions and the occasional mistake. Corrections are welcome.

## What you'll find here

- **Concept notes**: plain-language summaries of core ideas (sampling, spectrograms, pitch, prosody and more)
- **Jupyter notebooks**: hands-on experiments you can run, tweak and break
- **Papers and reading guides**: surveys and foundational papers, with notes on what to read first and what to skip
- **Personal notes**: rough thoughts, "aha" moments and things that confused me

## Repository structure

```
.
├── notes/              # Conceptual notes in Markdown, one topic per file (01-..., 02-...)
├── notebooks/          # Jupyter notebooks, named to match the notes
├── data/               # Small, legally usable sample audio (link to big datasets instead)
├── personal-notes/     # Informal, unpolished learning log
├── assets/             # Images and diagrams used in notes
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
└── README.md
```

## Learning roadmap

The path I'm following, assuming you already know a little Python and basic ML. It is a guide, not a requirement: you can start coding before you fully understand the maths, as long as you understand what a feature measures, roughly how it's computed, and when it might be unreliable.

### Phase 1: Audio fundamentals (about 3 days, 4-6 hours total)

| Day | Question | Learn | Task |
|-----|----------|-------|------|
| 1 | What is digital audio? | Sound waves and amplitude, frequency and pitch, sampling rate, mono vs stereo | Load a recording in a notebook, play it, plot the waveform, compute its duration. Resource: [librosa: loading audio](https://librosa.org/doc/main/auto_tutorials/01-intro/01-load.html) |
| 2 | How do we see what's inside sound? | Time vs frequency domain, intuitive Fourier transform, spectrograms and the STFT | Generate a spectrogram and point out where the sound gets stronger, weaker or changes frequency. Resource: [librosa: frequency and time-frequency analysis](https://librosa.org/doc/main/auto_tutorials/01-intro/02-frequency.html) |
| 3 | What makes speech sound different? | Fundamental frequency (F0), intensity and loudness, duration and pauses, voiced vs unvoiced sounds, prosody | Compare two spoken utterances and describe their delivery *before* extracting features. Resource: [Introduction to Prosody mini-tutorial](https://cs.utep.edu/nigel/intro-to-prosody/) (start with the intro videos, not all 29 lectures) |

**You're ready to move on if you can answer:** What does a waveform show? What does a spectrogram show? What is pitch (F0)? Why does sampling rate matter? What is prosody?

### Phase 2: See and measure speech (about 1 week)

1. **Learn to see speech**: use [Praat](https://praat.org/manual/Intro) to look at the waveform, spectrogram and pitch contour of a short recording while listening to it.
2. **Extract features in Python**: follow the [openSMILE Python docs](https://audeering.github.io/opensmile-python/) on a few short WAV files. Start with pitch, loudness and voicing rather than hundreds of features, and note where the measurements go wrong (silence, unvoiced speech, overlapping speakers).
3. **Write a tiny experiment note**: what you measured, what you observed, what you *cannot* conclude, and what to validate next.

Learning to extract pitch is a small technical exercise. Showing that a feature actually explains something is a research problem that needs proper controls, not just plots.

### Phase 3: Read your first papers

Don't read everything. Start with these, and focus first on the problem, diagrams, data, evaluation and limitations before the maths.

| Paper | Why read it |
|-------|-------------|
| [Deep Learning for Audio Signal Processing](https://arxiv.org/abs/1905.00078) (Purwins et al., 2019) | Broad survey of audio ML: representations, architectures, datasets, evaluation. A good orientation, not a step-by-step workbook. |
| [A Review of Deep Learning Techniques for Speech Processing](https://arxiv.org/abs/2305.00359) (Mehrish et al., 2023) | Speech-focused map covering MFCCs, architectures, representations, recognition and synthesis. Don't try to master every section. |
| [FastSpeech 2](https://arxiv.org/abs/2006.04558) (Ren et al., 2020) | A first conceptual paper on how duration, pitch and energy serve as explicit conditioning signals in speech synthesis. Read the model diagram and those sections first. |

Keep the full surveys as reference material, not homework.

### Phase 4: Build small things

1. Extract and visualise prosody (waveform, spectrogram, F0, intensity) on a short speech sample.
2. Build a simple baseline model, learning train/test splits, error metrics and uncertainty before adding complexity.
3. Add richer inputs and check whether improvements survive appropriate controls (for example speaker-disjoint splits).
4. Study an open-source implementation from a paper: read its README and data requirements, understand the inputs, labels, data flow and evaluation before trying to reproduce anything.

### What to skip at first

- Full audio and speech surveys
- Advanced speech-synthesis implementations
- Multiple feature-extraction toolkits at once
- Training a large neural network just to prove you've learned something

## Getting started

**Prerequisites:** Python 3.9+, basic Python, and some familiarity with NumPy. High-school maths is enough to begin.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter lab
```

Then pick a phase above, read the matching note in `notes/`, run the matching notebook in `notebooks/`, and change the parameters. Audio is best learned by ear.

## Contributing

Contributions are very welcome, from beginners and experts alike. Fixing a typo or suggesting a paper that helped you is as valuable as a new notebook.

- **Fix errors**: wrong explanations, broken code, dead links
- **Improve notes**: clearer explanations, better examples, diagrams
- **Add notebooks**: experiments on topics not yet covered
- **Suggest papers**: with a short note on why they help beginners
- **Open an issue**: questions, topic requests, or things that confused you

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and guidelines.

## Disclaimer

These are personal study notes, not authoritative teaching material. I try to be accurate, but I'm learning too. If something looks wrong, please open an issue or PR.

## License

Released under the [MIT License](LICENSE). Third-party audio, figures and papers referenced here keep their original licenses.

## Acknowledgements

Thanks to the authors, educators and open-source maintainers (librosa, Praat, openSMILE and others) whose work makes learning this field possible.
