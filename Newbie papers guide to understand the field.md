Yes — you should learn a few basic audio concepts first, but you do not need to complete a signal-processing course before starting practical work.

Since you already know ML, you have a useful foundation: features, representations, models, training and evaluation will feel familiar. What's new is understanding what the input signal physically represents and how we turn sound into numbers.

I'd put a tiny audio fundamentals bootcamp before the Praat and openSMILE exercises. Think of it as learning the vocabulary needed to understand your experiments, not taking another semester-long course.

## The concepts you need before CODA

## Your 3-day audio fundamentals bootcamp

Budget: around 4–6 hours total. No advanced maths required.

## Day 1

60–90 min

What is digital audio?

Learn:

- Sound waves and amplitude.
    
- Frequency and pitch.
    
- Sampling rate, samples and duration.
    
- Waveforms and channels (mono vs stereo).
    

Resource: [Librosa — Getting started with audio data](https://librosa.org/doc/main/auto_tutorials/01-intro/01-load.html). It loads, plays and plots real audio.

![](https://www.google.com/s2/favicons?domain=https://librosa.org&sz=32)

librosa 1.0.0 documentation

Your task: Load one recording in a Jupyter notebook, play it, plot its waveform and calculate its duration.

## Day 2

60–90 min

How do we see what's inside sound?

Learn:

- Time domain vs frequency domain.
    
- Fourier transform at an intuitive level.
    
- Spectrograms and the short-time Fourier transform (STFT).
    
- Why speech changes over time.
    

Resource: [Librosa — Frequency and time-frequency analysis](https://librosa.org/doc/main/auto_tutorials/01-intro/02-frequency.html).

![](https://www.google.com/s2/favicons?domain=https://librosa.org&sz=32)

librosa 1.0.0 documentation

Your task: Generate a spectrogram and identify where the sound becomes stronger, weaker or changes frequency.

## Day 3

90–120 min

What makes speech sound different?

Learn:

- Fundamental frequency (F0) and perceived pitch.
    
- Intensity and perceived loudness.
    
- Duration, speaking rate and pauses.
    
- Voiced vs unvoiced sounds.
    
- Prosody: how these features contribute to spoken delivery.
    

Resource: [Introduction to Prosody — Mini-Tutorial and Short Course](https://cs.utep.edu/nigel/intro-to-prosody/). Start with the introductory videos; don't try to complete all 29 lectures immediately.

![](https://www.google.com/s2/favicons?domain=https://www.cs.utep.edu&sz=32)

cs.utep.edu

Your task: Compare two spoken utterances and describe their delivery before extracting any features.

## How do you know you're ready to start implementation?

You don't need to master every concept. Check whether you can answer these:

What does a waveform show?What does a spectrogram show?What is pitch (F0)?Why does sampling rate matter?What is prosody?

One important note: you can begin coding before you fully understand the maths. You should, however, understand what a feature measures, how it is computed at a high level, and when it might be unreliable before using it as scientific evidence.

After these three days, move to Praat/openSMILE and your CODA pipeline. Keep the full audio survey and deeper speech papers for later, when you encounter concepts that require them.

This way, you're building just enough foundation to make your implementation meaningful without postponing your research for weeks.

Okay, let's simplify this dramatically. ❤️ You don't need another reading list. You need one starting point, one notebook, and one small result.

Because you're learning speech processing while developing CODA and preparing to contact X-LANCE and THUHCSI, I'd use this order.

## Your starting path

![Praat tutorial | Maria Gouskova](https://images.openai.com/static-rsc-4/Xxkj4ftxMtGiDJBJdQQPygn9KysQgvyZfLr6VMy0zhbqfCGaxYOKLzKg0RNCG4B_k_fhMQUcKsj1jtk4d6e-skfHo56sGaQMDGs_E-bDfDZdVvSSymmHlvvas6hZtL8LkzRC7brdOsm4LAuOApPBh2xhy7lAkj-kwagy6a2LyXM?purpose=inline)

Start here

1. Learn to see speech

Day 1 · 60–90 minutes

Open the [Praat introduction](https://praat.org/manual/Intro). Learn just three things: waveform, spectrogram and pitch contour. Listen to a short recording while looking at its plots. You don't need to read the entire manual.

![](https://www.google.com/s2/favicons?domain=https://praat.org&sz=32)

praat.org

![Let Me Visualize Your Words, and I’ll Tell You Who You Are – Testing Site 2](https://images.openai.com/static-rsc-4/RxigWFekhIyR2W4htrW2-sm572sUDdpW3e27NfqZBW2hs7FSzdHwdIY6vl-HdcuG3cePA53YxCLicappqmvsQRaCtbNjywQLaNlj4VUVgFuu4O9aCYtB--dxNGtaA_Ke8XoueCp9TpVgcWi76lKVMPsZhXlqRCYtIUa2ybJadXw?purpose=inline)

2. Extract features using Python

Day 2 · 1–2 hours

Follow the [openSMILE Python documentation](https://audeering.github.io/opensmile-python/). Extract a small set of acoustic features from a few recordings and plot them. Start with pitch, loudness and voicing rather than hundreds of features.

![](https://www.google.com/s2/favicons?domain=https://audeering.github.io&sz=32)

Documentation

+1

![Bike | How Much does Prosody Help Turn-taking?](https://images.openai.com/static-rsc-4/5x8KnJmq74b57aDDWmsyuAiXHNXehBiyAuudygGZUG0g8z4LIMk8XjzyjEpSUn6FK3DEKltABqS8lu3R-1qe5zlXdY_12utVQ97FHKKBuzpIP9KHRux9BXUNFRgXebqzuE1wVbqt1WUBzkPWnx-4QvJrHaTLDBb71L4INmgkbz8?purpose=inline)

3. Connect it to CODA

Days 3–4 · 2–4 hours

For a few conversations, compare the acoustic delivery of one speaker's turn with the next speaker's turn. Ask: Can I measure the delivery consistently? What might explain the difference? This is a learning exercise, not evidence that CODA's hypothesis is true.

![Speech Synthesis](https://images.openai.com/static-rsc-4/NjBfMTSr_fTlmK4adsYwXQ7iAlh9JIQtmlWIUkRzQtOPRgoJxbROqi7Ed1qkuBjhygeFaYhc8VIpdlwSZpWvRch6I74lLhNdRU-30DQiygzu2NIDraLCcksgaGyhqwWea3zu2KDGx8M1TwFFhXdQTm8AawrO53wMkmhrBLQ2eWQ?purpose=inline)

4. Read just one research paper

Day 5 · 1–2 hours

Read the relevant parts of [FastSpeech 2](https://arxiv.org/abs/2006.04558) to understand how duration, pitch and energy are used in speech synthesis. Then move to the THUHCSI paper on contextual speaking styles.

## What to ignore for now

- The full audio and speech surveys.
    
- Advanced speech synthesis implementations.
    
- Multiple feature-extraction toolkits.
    
- Reading every X-LANCE and THUHCSI paper.
    
- Training a large neural network just to prove you've learned something.
    

Keep the surveys bookmarked as reference material, not homework.

## Your first four sessions

Your checklist

0 of 4

Open one short speech recording

Listen to it and inspect the waveform and spectrogram in Praat.

Understand three acoustic measurements

Learn pitch (F0), loudness/intensity and voicing. Write one sentence explaining each.

Extract and plot features in Python

Use openSMILE on a few short WAV files; inspect the output and note possible measurement errors.

Write a tiny CODA experiment note

Record what you measured, what you observed, what you cannot conclude and what you need to validate next.

One important distinction: learning to extract pitch is a small technical exercise. Establishing that one speaker's prosody predicts another's beyond lexical content and other confounds is the actual research problem—and it requires your registered controls and evaluation, not just plots.

My advice is to start with Session 1 only. Don't open another paper or tutorial until you've looked at your first speech waveform and spectrogram. That's your first concrete milestone.
## 3. Learn the concepts before the advanced papers

Since you know basic ML but are new to speech research, I'd use these as a foundation. You do not need to understand every equation on the first pass.

![PPT - Introduction to text-to-speech synthesis PowerPoint Presentation - ID:4242883](https://images.openai.com/static-rsc-4/XLoGp0YhMf6tZZRQdFmSOM2TRfnAVxlC_YxWhyjDw_9aXiho83Mh1cp8mKJQgWP0KODW-__ALWPL_BpJS_JxH6ScFBLgooJbG56R0mDxVCig8LNC2KnpGzl1QhzEwBOqC12k30R8Ae78SqXXUb311MlSg6Y6JlaTIk0BQguzPnk?purpose=inline)

A. Understanding prosody and acoustic features

Start with the [openSMILE toolkit and documentation](https://github.com/audeering/opensmile).

Learn what pitch/F0, intensity, duration, pauses, voicing, and spectral features measure. Practise extracting features from short speech recordings and inspecting their distributions.

Practical resource rather than a conceptual paper. Especially useful for understanding the measurement stage of CODA.

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

![Speech Synthesis](https://images.openai.com/static-rsc-4/NjBfMTSr_fTlmK4adsYwXQ7iAlh9JIQtmlWIUkRzQtOPRgoJxbROqi7Ed1qkuBjhygeFaYhc8VIpdlwSZpWvRch6I74lLhNdRU-30DQiygzu2NIDraLCcksgaGyhqwWea3zu2KDGx8M1TwFFhXdQTm8AawrO53wMkmhrBLQ2eWQ?purpose=inline)

B. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)

Ren et al., 2020.

This is a useful first conceptual paper for seeing how duration, pitch, and energy can be represented as explicit conditioning signals in a speech-generation model.

Read the model diagram and the sections introducing duration, pitch, and energy first. You can skip most implementation details on your first pass.

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

![ProsoSpeech: Enhancing Prosody With Quantized Vector Pre-training in Text-to-Speech](https://images.openai.com/static-rsc-4/YIPC35yCFNOTlI9chBrQVYsWISmvWSYPh5uNYmWUnBraXL2CuSu4GzCekumAzqaLnzA5FIKE5G4BA7p_bFljIQVPYjNkZ-WAYGG10-_xQ7dThHU0Zlj97QW3heSIjA4kNGIQenhBJy6ctF1vgzfxo2GSJSzk7WB7dSE7gJmhJMU?purpose=inline)

C. [Controllable Speech Synthesis by Learning Discrete Phoneme-Level Prosodic Representations](https://doi.org/10.1016/j.specom.2022.11.006)

Speech Communication, 2023.

This introduces the idea of learning discrete prosody representations from pitch and duration and using them to control synthesis. It is useful for understanding the difference between measuring prosody, representing prosody, and generating speech conditioned on prosody.

![](https://www.google.com/s2/favicons?domain=https://www.sciencedirect.com&sz=32)

ScienceDirect

## 4. The best way to learn through implementation

You don't need to train a speech-generation model from scratch. Start with a small notebook and build up.

## Your beginner practical sequence

1

Extract and visualise prosody

Use a short, legally usable speech sample. Plot waveform, spectrogram, pitch/F0, and intensity. Learn what happens when a pitch tracker encounters silence, unvoiced speech, or overlapping speakers.

Resource: [openSMILE](https://github.com/audeering/opensmile) and its documentation.

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

2

Build a simple context-only baseline

Represent each turn with a few defensible contextual variables and predict a prosodic target for the next turn. Learn train/test splits, regression, error metrics, and uncertainty before adding complex models.

3

Add previous-speaker prosody

Compare the baseline with a model that additionally uses the previous speaker's prosodic features. Use speaker/dyad-disjoint splits where required by your research design, and check whether any apparent improvement survives appropriate controls.

4

Study THUHCSI's implementation

Explore the lab's [public conversational TTS repository](https://github.com/thuhcsi/mm2022-conversational-tts). It contains code for the multimodal conversational speaking-style paper, including training and testing scripts. Read the README and data requirements before attempting to run anything.

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

thuhcsi/mm2022-conversational-tts · GitHub

I would not start by reproducing the complete THUHCSI graph model. First understand the inputs, labels, data flow, and evaluation. The repository is useful as an implementation reference, but reproducing it is a much larger task than building your first CODA experiment.

## 5. A manageable reading plan before your emails

### 7-day checklist

0 of 7

Day 1

Speech and prosody basics

Learn pitch, intensity, duration, pauses, voicing, and spectrograms. Inspect a few recordings.

Day 2

FastSpeech 2

Understand how duration, pitch, and energy are used as conditioning signals.

Day 3

Prosody representation

Read the discrete prosodic representation paper and understand feature extraction versus synthesis control.

Day 4

THUHCSI conversational style

Read the 2022 paper and inspect its public repository and model diagram.

Day 5

THUHCSI contextual dependencies

Read the 2025 paper. Focus on what contextual information is used and how the system is evaluated.

Day 6

X-LANCE prosody modelling

Read the prosody-tagging paper and connect it to controllable speech generation.

Day 7

Write your research-fit notes

Write 3–5 sentences per lab explaining one idea you understood, how it connects to your work, and what you want to investigate.

Seven days is a learning target, not a requirement to master the field. If a paper is difficult, focus on its problem, diagram, data, evaluation, and limitations before working through the mathematics.

My advice: start with the speech-feature notebook, then FastSpeech 2, then the THUHCSI contextual-speaking-style paper. That order will help you understand what the papers are doing instead of merely recognising their terminology. You can mention a paper in an email once you can explain its central idea in your own words.

Ahh, got it — you don't just want beginner tutorials. You want the audio-processing equivalent of that mechanistic interpretability review: one curated paper that explains the foundations, organizes the field, points to important papers, and ideally gives you a roadmap from beginner concepts to actual research.

Your linked paper explicitly does this through its beginner's roadmap, practical workflows and case studies.

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

I looked for the closest equivalents in audio and speech research.

## Best matches for what you're looking for

![Special Issue Machine Learning for Audio Published in IEEE JSTSP](https://images.openai.com/static-rsc-4/NJg-mSufgeoQFWCLlqqpDcWPC1dRjnCs8p8ZwWY3YQdrHgDcBTsHi21FHiGKEYMarp8T-mUK-NsAUSPodQ1ENXIJLVgiWltcjzygKzhSzVkiSk9AmD3pl1eZyHXmYXGk6oqZEFQ2BmyRxrKA4r9Rqekvwpu9Pu8EXWzy34pb0go?purpose=inline)

Best overall starting survey

1. [Deep Learning for Audio Signal Processing](https://arxiv.org/abs/1905.00078)

Purwins et al. · 2019 · Review paper

Probably the closest match if you want one paper to orient you across audio ML. It covers audio representations, model architectures, datasets, evaluation, recognition, synthesis and enhancement.

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

Why read it: It helps you understand the overall audio ML landscape before specializing in speech.

Limitation: It is a broad technical survey, not a step-by-step beginner workbook.

![Client Challenge](https://images.openai.com/static-rsc-4/usymlS_-0OMeB2lN0HTsRuSehNHM3G6ygbhxleJ5TlsMzPnSBK3_NCDuSPh8wO_XgbU-zA_efc1e7ULearkFcDILdNZqKq5LosUmMkwF50MmBq1WE0VCQOO4e_CpwmeIjFRTjBSQNqIN6hOh1G6hfVYBgT1ma7gPxIzlKNfNubY?purpose=inline)

2. [A Review of Deep Learning Techniques for Speech Processing](https://arxiv.org/abs/2305.00359)

Mehrish et al. · 2023 · Review paper

More speech-focused: it covers traditional features such as MFCCs, deep-learning architectures, speech representations, speech-recognition and synthesis tasks, and transfer learning.

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

Why read it: It gives you a broader map of speech ML and helps connect your existing ML knowledge to speech applications.

Limitation: It covers many tasks, so don't attempt to master every section on your first pass.

![Prosody Research and Applications The State of the](https://images.openai.com/static-rsc-4/5wwQM_8FkltVDFjRpy7CKJRKftfhNS1SD9sJO4A7LNw8lZtLlMbyQMS9ZUjFo1QM2Zu_QyuXQjuF8IYQuCzHqP0XaUAoSx4o30z_i2cdR2gQDlU_98r3hspVqzhAUSQvFUMKM5fiq62MYrJ0XMRWLrG7rpBIUq6wscsYkeyMP2U?purpose=inline)

Best for CODA specifically

3. [Introduction to Prosody: A Mini-Tutorial and a Short Course](https://cs.utep.edu/nigel/intro-to-prosody/)

Prosody tutorials and course materials

This is especially promising for you: it is explicitly accessible without prior phonetics knowledge and includes lecture materials, a starter bibliography, audio examples and hands-on analysis with tools such as Praat. It discusses prosody in dialogue, turn-taking, speech synthesis and speaker-state inference.

![](https://www.google.com/s2/favicons?domain=https://cs.utep.edu&sz=32)

cs.utep.edu

Why read it: It gets you closer to understanding what CODA is measuring and why those measurements matter.

Limitation: It is a curated teaching resource rather than one comprehensive audio deep-learning paper.

## How I'd use these for your situation

You don't need to read three full surveys cover to cover. I'd use them as a small, curated curriculum:

Step 1 — Understand sound and prosody

Start with the prosody course. Focus on pitch, intensity, timing, pauses, voicing and how delivery changes in conversation.

Step 2 — Learn the audio representations

Read the audio-processing survey's sections on problem types, features and evaluation. Learn what waveforms, spectrograms, log-mel features and MFCCs actually represent.

Step 3 — Connect to speech research

Use the speech-processing review to explore speech representations and the tasks relevant to X-LANCE and THUHCSI. Then read the lab papers on conversational speaking style.

Step 4 — Implement while learning

Use [openSMILE](https://github.com/audeering/opensmile) or Praat to analyze a few recordings. Plot features, inspect failures and ask what each feature can—and cannot—tell you.

My verdict: If you want one starting point for general audio ML, choose Deep Learning for Audio Signal Processing. If your immediate priority is CODA and understanding the lab papers, start with the prosody mini-tutorial instead.

I didn't find a single resource in this search that matches the linked MI review in all respects—beginner roadmap, detailed method taxonomy, implementation workflows and case studies. These three together get you closer without overwhelming you with unrelated material.