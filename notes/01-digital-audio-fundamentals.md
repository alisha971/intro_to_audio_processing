# Digital Audio Fundamentals

## Overview

Digital audio is sound represented as a sequence of numbers that a computer can store and process. Sound begins as a physical vibration. A microphone converts the vibration into an electrical signal, and an analogue-to-digital converter measures that signal at regular intervals and encodes the measurements numerically.

1. **Physical sound**: pressure variations travel through air.
2. **Microphone**: converts pressure variations into an electrical signal.
3. **Sampling and quantization**: the signal is measured at discrete times and the measurements are stored at finite numerical precision.
4. **Digital audio**: a sequence of numbers that software can analyse.

Speech recognition, voice agents and audio-language models do not receive sound waves directly. They receive a digital representation of sound, usually transformed into other representations (for example spectrograms) before processing.

This note covers five foundational concepts:

- Sound waves and amplitude
- Frequency and pitch
- Sampling rate, samples and duration
- Waveforms
- Mono and stereo audio

## 1. Sound waves and amplitude

Speech is produced when the vocal folds and vocal tract create pressure variations in the surrounding air. These variations propagate as sound waves. A microphone placed in the sound field registers the changing pressure over time and outputs a corresponding signal.

**Amplitude** is the magnitude of a signal's variation. On a graph it is the height of the wave. Small amplitude is generally associated with quieter sound and large amplitude with louder sound.

In digital audio, amplitude is a numerical value at each sample. In normalised floating-point audio, values conventionally lie between $-1$ and $+1$:

```python
audio = [0.02, 0.15, 0.42, 0.18, -0.10, -0.35]
```

Each value is the amplitude of the signal at one instant. Amplitude can be positive or negative; the sign indicates on which side of the signal's reference (zero) level the value lies. It does not indicate negative sound.

Amplitude is related to loudness but is not identical to it. Perceived loudness also depends on frequency, duration and the frequency response of the human ear.

## 2. Frequency and pitch

Amplitude describes the size of a wave; **frequency** describes how rapidly it repeats. Frequency is measured in hertz (Hz), where 1 Hz is one cycle per second.

- $100\text{ Hz}$ is 100 cycles per second.
- $440\text{ Hz}$ is 440 cycles per second.
- $1{,}000\text{ Hz}$ is 1,000 cycles per second.

A low-frequency wave completes few cycles in a given time and is heard as a low pitch. A high-frequency wave of the same amplitude completes more cycles in the same time and is heard as a higher pitch. A bass guitar typically produces lower-pitched sounds than a whistle.

**Pitch versus frequency.** Frequency is a measurable physical property of a signal. Pitch is the perceptual sensation associated with it. Real speech and musical sounds contain many frequencies at once, so the two are closely related but not identical.

### Period

The time taken for one complete cycle is the **period**, $T$, and is the reciprocal of frequency:

$$T=\frac{1}{f}$$

where $f$ is the frequency in hertz and $T$ is the period in seconds.

For a 100 Hz signal:

$$T=\frac{1}{100}=0.01\text{ seconds}$$

One cycle lasts 10 milliseconds.

For a 1,000 Hz signal:

$$T=\frac{1}{1000}=0.001\text{ seconds}$$

One cycle lasts 1 millisecond. Higher frequency therefore means a shorter period.

### Relevance to speech

Speech carries information in its frequency structure:

- Pitch contributes to intonation and to vocal characteristics.
- Frequency patterns distinguish speech sounds from one another.
- Changes in pitch over time can signal questions, emphasis and emotion.

## 3. Sampling rate, samples and duration

Physical sound varies continuously in time, but a computer cannot store every instant of a continuous signal. A digital recording therefore measures the signal at discrete, evenly spaced times. Each measurement is a **sample**.

The **sampling rate** ($f_s$) is the number of samples recorded per second.

| Sampling rate        | Meaning               | Time between samples    |
| -------------------- | --------------------- | ----------------------- |
| 8,000 Hz (8 kHz)     | 8,000 samples/second  | 125 microseconds        |
| 16,000 Hz (16 kHz)   | 16,000 samples/second | 62.5 microseconds       |
| 44,100 Hz (44.1 kHz) | 44,100 samples/second | About 22.7 microseconds |
| 48,000 Hz (48 kHz)   | 48,000 samples/second | About 20.8 microseconds |

Speech-processing systems commonly use 16 kHz. Music and general audio are typically stored at 44.1 kHz or 48 kHz.

A higher sampling rate gives more measurements per second but does not by itself guarantee a better recording. Quality also depends on the microphone, background noise, quantization and other factors.

### Number of samples

The number of samples in a recording is the product of the sampling rate and the duration:

$$N=f_s\times t$$

where $N$ is the number of samples, $f_s$ is the sampling rate in samples per second and $t$ is the duration in seconds.

A 3-second recording at 16,000 Hz contains

$$N=16{,}000\times3=48{,}000$$

samples per channel. A 10-second recording at 16 kHz contains 160,000 samples per channel.

### Quantization

Sampling determines *when* the signal is measured. **Quantization** determines how each measured amplitude is represented using a finite set of numerical levels.

Standard 16-bit PCM audio, for example, can represent $2^{16}=65{,}536$ distinct integer amplitude levels.

| Concept      | What it controls                                                   |
| ------------ | ------------------------------------------------------------------ |
| Sampling     | Measurement of the signal at particular points in time             |
| Quantization | Numerical precision with which each measured amplitude is stored   |

Both concepts apply to WAV files and to audio-processing libraries.

## 4. Waveforms

A **waveform** is a graph of a signal's amplitude against time. It is one of the most common ways to inspect digital audio.

- The horizontal axis represents time.
- The vertical axis represents amplitude.
- The individual samples together form the digital signal.

Real speech waveforms are irregular because speech combines many frequencies and changes continuously. A waveform can reveal:

- Silence or pauses between words.
- Loud and quiet sections.
- Clipping, where the signal exceeds the limits of the recording system.
- Speech segments that may need to be extracted from a longer recording.

A waveform does not show which words were spoken. Identifying words requires further analysis, for example of the signal's frequency content.

### Example: generating a synthetic signal

The following code generates one second of a 440 Hz sine wave sampled at 16 kHz and plots the first 20 milliseconds. No microphone or audio file is required.

```python
import numpy as np
import matplotlib.pyplot as plt

# Audio configuration
sample_rate = 16000
duration = 1.0
frequency = 440

# Sample times
time = np.arange(int(sample_rate * duration)) / sample_rate

# 440 Hz sine wave
audio = 0.5 * np.sin(2 * np.pi * frequency * time)

# Plot the first 20 milliseconds
plt.figure(figsize=(10, 4))
plt.plot(time[:320], audio[:320])
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Digital Audio Waveform: 440 Hz")
plt.grid(True)
plt.show()

print("Number of samples:", len(audio))
print("Sampling rate:", sample_rate)
print("Duration:", len(audio) / sample_rate)
```

- `sample_rate = 16000` sets 16,000 samples per second.
- `np.arange(...) / sample_rate` produces the timestamp of each sample.
- The sine expression produces a repeating signal with a frequency of 440 Hz.
- `audio` holds the numerical amplitude values.
- The factor `0.5` limits the amplitude to the range $-0.5$ to $+0.5$.

This is a pure tone rather than human speech. It is used because its properties are simple to verify: 16,000 samples, a 1.0 s duration, and a period of about 2.27 ms.

## 5. Mono and stereo audio

Audio **channels** are separate signal streams that are recorded or played back together.

- **Mono** has one channel: a single sequence of samples represents the recording. Examples include many voice recordings and telephone speech.
- **Stereo** has two channels: one sequence for the left and one for the right. This preserves spatial differences, such as a source appearing nearer to one side.

### Effect on sample count

For a 5-second recording sampled at 16 kHz:

| Property            | Mono      | Stereo    |
| ------------------- | --------- | --------- |
| Sampling rate       | 16,000 Hz | 16,000 Hz |
| Channels            | 1         | 2         |
| Samples per channel | 80,000    | 80,000    |
| Total sample values | 80,000    | 160,000   |

Stereo has 80,000 samples per channel but 160,000 sample values in total. With the same bit depth in both channels, uncompressed stereo PCM requires twice the data of mono at the same duration and sampling rate.

### Representation in Python

A mono signal is typically a one-dimensional array and a stereo signal a two-dimensional array:

```python
mono_audio.shape    # (80000,)
stereo_audio.shape  # (80000, 2)
```

The exact layout depends on the library and file format. Some libraries place channels first, giving `(2, 80000)`. Checking an array's shape is therefore one of the first steps after loading a recording.

## Summary

| Term          | Definition                                                         |
| ------------- | ------------------------------------------------------------------ |
| Amplitude     | Size of the signal's variation; related to loudness                |
| Frequency     | Number of cycles per second; related to pitch                      |
| Period        | Duration of one cycle, $T = 1/f$                                   |
| Sampling rate | Number of amplitude measurements per second                        |
| Quantization  | Representation of each measurement at finite numerical precision   |
| Waveform      | Graph of amplitude against time                                    |
| Channels      | Number of parallel signal streams: one for mono, two for stereo    |

Key relation: $N = f_s \times t$ samples per channel.

## Review questions

1. A recording is 2 seconds long and sampled at 24,000 Hz. How many samples does it contain per channel? (48,000)
2. A signal has a larger amplitude. What does this usually suggest? (Greater loudness, not higher pitch)
3. What does a 16 kHz sampling rate mean? (16,000 samples per second)
4. How many samples per channel are in 4 seconds at 16 kHz? (64,000)
5. What is the main difference between mono and stereo? (Mono has one channel; stereo has two)

## References

- librosa documentation, *Getting started with audio data*: <https://librosa.org/doc/main/auto_tutorials/01-intro/01-load.html>
- Aalto University, *Introduction to Speech Processing*: <https://speechprocessingbook.aalto.fi/>
