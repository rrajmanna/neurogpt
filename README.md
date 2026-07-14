# neurogpt

A pipeline for detecting movement in intracranial EEG (iEEG) using a transformer trained to predict the next segment of neural signals.

## Dataset

[OpenNeuro ds005931](https://openneuro.org/datasets/ds005931): intracranial EEG from pediatric epilepsy patients, recorded during seizure monitoring. While electrodes were implanted, patients played a finger-tapping game ("Speed Match" on Lumosity).

This project uses one subject, one session: 64 electrodes, 7 minutes of recording, 181 finger-tap events.

## Pipeline

1. **Load and clean**: read the  EDF file, keep only electrodes marked "good" (excluding ones flagged as noisy), apply standard filtering
2. **Segment into windows**: split the recording into 200ms chunks, each labeled "movement" or "rest" based on proximity to a finger-tap event.
3. **Extract features**: in addition to raw voltage, compute band power (theta, alpha, beta, gamma) for each channel and window
4. **Train the model**: a transformer that takes a sequence of windows and learns to predict the next one, trained with loss rather than direct regression.
5. **Analyze**: extract the model's learned embeddings for movement vs. rest windows and check whether they separate.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy scipy scikit-learn matplotlib jupyterlab mne datalad
brew install git-annex

datalad clone https://github.com/OpenNeuroDatasets/ds005931.git data/ds005931
cd data/ds005931
datalad get sub-01/ses-02/ieeg/*.edf
cd ../..
```
