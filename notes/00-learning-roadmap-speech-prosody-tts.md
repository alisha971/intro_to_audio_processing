# Speech, Prosody and Text-to-Speech: A Conceptual Map

## Overview

This note maps the field that connects speech signal analysis, prosody and neural text-to-speech (TTS). It defines the main concepts, lists the standard tools and foundational papers, and orders them from signal basics to modern expressive synthesis.

The guiding question is: *which information in speech represents how something is said, as opposed to what is said or who says it?*

## 1. Components of a speech signal

A speech signal can be decomposed, approximately, into four kinds of information:

| Component                | Describes             | Examples                                           |
| ------------------------ | --------------------- | -------------------------------------------------- |
| Linguistic content       | What was said         | Words, phonemes, syntax                            |
| Speaker identity         | Who said it           | Vocal tract characteristics, timbre                |
| Prosody                  | How it was said       | Pitch, energy, duration, rhythm, pauses            |
| Paralinguistic behaviour | What else is conveyed | Emotion, hesitation, laughter, breath, turn-taking |

```text
                     SPEECH
                       │
          ┌────────────┴────────────┐
          │                         │
     LINGUISTIC                 PARALINGUISTIC
      CONTENT                    / ACOUSTIC
          │                         │
       words                    ┌────┴─────┐
       phonemes                 │          │
       syntax                Speaker    Behaviour
                              identity    / style
                                           │
                              ┌────────────┼────────────┐
                              │            │            │
                            Pitch       Energy       Timing
                              │            │            │
                           intonation   emphasis    pauses
                                                    speech rate
                                                    rhythm
```

### Prosody

**Prosody** is the set of suprasegmental properties of speech: those that extend over units larger than a single sound. It includes:

- Pitch, measured as fundamental frequency (F0), and the pitch contour over time
- Intensity (energy)
- Duration and speech rate
- Rhythm and pauses
- Stress and emphasis
- Boundary tones and intonation

Prosody is not the same as emotion. Emotion is one of several things that prosody can express. Broader behavioural characteristics of speech include hesitations, fillers, laughter, breath, interruptions, backchannels, response latency, turn-taking and conversational rhythm.

## 2. Signal fundamentals

The following concepts are the prerequisites for speech analysis. Digital-audio basics are covered in [01-digital-audio-fundamentals.md](01-digital-audio-fundamentals.md).

- Waveform, sampling rate, amplitude, frequency
- Fourier transform, short-time Fourier transform (STFT) and spectrogram
- Mel spectrogram
- Fundamental frequency (F0) and harmonics
- Formants
- Energy and intensity
- Voiced versus unvoiced speech

A typical analysis chain is:

```text
waveform → spectrogram → F0 contour → energy contour
```

Each stage represents a different view of the same signal: amplitude over time, frequency content over time, perceived pitch over time and loudness over time.

**Tool: librosa.** [librosa](https://github.com/librosa/librosa) is a Python library for audio and music analysis. It covers loading audio, waveforms, STFT, spectrograms, mel spectrograms, RMS energy and pitch-related analysis.

## 3. Speech analysis tools

### Praat and Parselmouth

[Praat](https://www.fon.hum.uva.nl/praat/) is the standard phonetics tool. [Parselmouth](https://github.com/YannickJadoul/Parselmouth) exposes Praat's analysis functions in Python.

| Feature            | Interpretation                                |
| ------------------ | --------------------------------------------- |
| F0                 | Pitch                                         |
| F0 mean            | Typical pitch                                 |
| F0 range           | Pitch expressiveness                          |
| F0 variance        | Pitch variability                             |
| Energy             | Loudness or intensity                         |
| Duration           | Length of sounds or words                     |
| Pause duration     | Hesitation and rhythm                         |
| Speech rate        | Speed including pauses                        |
| Articulation rate  | Speed excluding pauses                        |
| Formants           | Vocal tract characteristics                   |
| Voice quality      | Breathiness, roughness and related properties |

### openSMILE

[openSMILE](https://github.com/audeering/opensmile-python) extracts acoustic features and is widely used in emotion recognition, affective computing and paralinguistics. It provides standardised feature sets, including GeMAPS and eGeMAPS, a compact set covering F0, energy, spectral features, voice quality and temporal features. The larger ComParE set contains over 6,000 features and is generally used for broad benchmarking rather than interpretation.

### Forced alignment

**Forced alignment** takes audio and its transcript and produces time stamps for each word and phoneme:

```text
I          really       don't       think       that's
|-----------|-------------|-----------|-----------|
0.00       0.31          0.74        1.02        1.35
```

With alignment, one can compute word and phoneme durations, pauses before and after words, local speaking rate, and the pitch and energy associated with particular words. The [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) is a widely used open-source implementation.

### Other toolkits

- [SpeechBrain](https://github.com/speechbrain/speechbrain): PyTorch toolkit with recipes and pretrained models for ASR, speaker recognition and speech processing.
- [ESPnet](https://github.com/espnet/espnet): large end-to-end toolkit covering TTS, ASR, vocoders and pretrained systems. It is a broad reference rather than an introductory codebase.

## 4. The TTS pipeline

The classic neural TTS architecture has two stages. An acoustic model maps text to an intermediate acoustic representation, and a vocoder converts that representation to a waveform.

```text
Text → Phonemes → Acoustic model → Mel spectrogram → Vocoder → Waveform
```

The mel spectrogram is used as the intermediate representation because it is compact, it approximates human frequency perception, and it is much easier for a model to predict than raw samples.

### Tacotron 2

*Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* ([arXiv:1712.05884](https://arxiv.org/abs/1712.05884)). An encoder-attention-decoder network predicts mel spectrograms from text, and a WaveNet vocoder converts them to audio. Reference implementation: [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2).

### WaveNet

WaveNet established neural waveform generation by modelling raw audio sample by sample. In early neural TTS it served as the vocoder that turns a mel spectrogram into a waveform. See [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) for an open implementation.

## 5. Prosody in TTS: the one-to-many problem

The mapping from text to speech is **one-to-many**. The same sentence can be realised with different pitch, rhythm, duration, emphasis, energy, pauses and speaking style, depending on emotion, intention and situation.

```text
             SAME TEXT
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
      happy    angry    bored
        ↓        ↓        ↓
    different acoustic realisations
```

### FastSpeech 2

*FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* ([arXiv:2006.04558](https://arxiv.org/abs/2006.04558)). FastSpeech 2 addresses the one-to-many problem by conditioning on explicit variation information: **duration**, **pitch** and **energy**. These are predicted from the phoneme representation and added before the mel-spectrogram decoder.

```text
TEXT → phoneme representation
          ├── content
          └── prosody: duration, F0, energy
                    ↓
              mel spectrogram → vocoder → audio
```

Implementation: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). Its documentation notes that phoneme-level pitch and energy prediction can improve prosody.

## 6. Style modelling

### Global Style Tokens

*Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis* ([arXiv:1803.09017](https://arxiv.org/abs/1803.09017)). A reference encoder compresses an utterance into a style embedding, which attends over a bank of learned **Global Style Tokens**. The tokens capture expressive variation without explicit style labels. They can control speaking style and speed and can transfer style from a reference utterance.

```text
speech → reference encoder → style representation → GSTs → Tacotron → speech
```

### Text-Predicted Global Style Tokens

*Predicting Expressive Speaking Style From Text in End-to-End Speech Synthesis* ([arXiv:1808.01410](https://arxiv.org/abs/1808.01410)). This work asks whether a latent style can be predicted from text alone. It distinguishes **reference-conditioned style** (style copied from an audio example) from **text-conditioned style** (style inferred from the input text).

## 7. Modern expressive TTS

### VITS

*Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech* ([arXiv:2106.06103](https://arxiv.org/abs/2106.06103)). VITS combines a conditional variational autoencoder, normalising flows and adversarial training in a single end-to-end model. Its **stochastic duration predictor** models the fact that one text can be spoken with different rhythms and pitches. Speech behaviour is therefore treated as a distribution rather than a single deterministic output:

```text
text + situation + speaker state → distribution over behaviours
```

### StyleTTS 2

*StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models* ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691); code: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)). StyleTTS 2 models style as a latent random variable sampled by diffusion and uses WavLM, a large pretrained speech model, as a discriminator feature extractor in adversarial training.

```text
text → linguistic representation
          ├── content
          └── style → diffusion → latent style
                          ↓
                      synthesis
```

## 8. Concept dependency order

The topics above build on one another in the following order:

```text
Speech fundamentals
  waveform, sampling, STFT, spectrogram, mel spectrogram,
  F0, energy, formants, voiced/unvoiced
        ↓
Speech analysis
  Praat/Parselmouth, openSMILE, prosodic features,
  temporal features, voice quality, forced alignment
        ↓
TTS foundations
  Tacotron 2, WaveNet and vocoders, FastSpeech 2
        ↓
Style
  Global Style Tokens, Text-Predicted GST, style representations
        ↓
Modern TTS
  VITS, StyleTTS 2
        ↓
Behavioural modelling
  behavioural features, prosody representation,
  speaker/style disentanglement, situational conditioning, style transfer
```

## 9. Representation view

A behavioural-modelling pipeline separates content from behaviour, and behaviour from speaker:

```text
                 SPEECH
                    │
          ┌─────────┴─────────┐
       CONTENT             BEHAVIOUR
     phonemes             F0, energy, timing
     words                     │
     semantics         behavioural representation
                               │
                  ┌────────────┴────────────┐
              SITUATION                 SPEAKER
                  └────────────┬────────────┘
                               ↓
                        predicted style → TTS
```

Open questions in this framing include where a model represents pitch, pause, emphasis and speaking-rate behaviour, whether latent dimensions correspond to behavioural style, and whether the representation of a speaking attitude differs from the representation of speaker identity.

## Reference list

| Resource                  | Type    | Link                                                       |
| ------------------------- | ------- | ---------------------------------------------------------- |
| librosa                   | Library | <https://github.com/librosa/librosa>                       |
| Parselmouth               | Library | <https://github.com/YannickJadoul/Parselmouth>             |
| openSMILE (Python)        | Library | <https://github.com/audeering/opensmile-python>            |
| Montreal Forced Aligner   | Tool    | <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner> |
| SpeechBrain               | Toolkit | <https://github.com/speechbrain/speechbrain>               |
| ESPnet                    | Toolkit | <https://github.com/espnet/espnet>                         |
| Tacotron 2                | Paper   | <https://arxiv.org/abs/1712.05884>                         |
| FastSpeech 2              | Paper   | <https://arxiv.org/abs/2006.04558>                         |
| Global Style Tokens       | Paper   | <https://arxiv.org/abs/1803.09017>                         |
| Text-Predicted GST        | Paper   | <https://arxiv.org/abs/1808.01410>                         |
| VITS                      | Paper   | <https://arxiv.org/abs/2106.06103>                         |
| StyleTTS 2                | Paper   | <https://arxiv.org/abs/2306.07691>                         |
