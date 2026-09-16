# %% [markdown]
# # Classification Track : BCI Motor Imagery
#
# This notebook contains the classification pipeline for classifying 4 motor imagery tasks - left hand, right hand, tongue, feet using traditional machine learning techniques. We use the BCI Competition IV 2a dataset provided by Institute for Knowledge Discovery, Graz University of Technology.
#
# Unlike traditional ML pipelines, this project requires us to do signal processing + feature processing to develop a tabular dataset. The EEG data is time series in nature, and comes along with markers explaining when a particular task was imagined.
#
# ![image.png](attachment:image.png)
#
# The above is the timing diagram of the experiment. Before the ML pipeline, we shall epoch each of these events, and apply processing techniques like Common Spatial Pattern (CSP) to obtain numerical features, which will be fed to different classifiers in the pipeline.

# %%
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mne.set_log_level('ERROR')

# %% [markdown]
# ### Section A - Dataset and EDA

# %% [markdown]
# The competition provides two datasets for each subject. In our case there are 9 subject recordings, however we shall limit to only using the first subject's data for this track. One dataset is supposed to be trained on fully, and should be tested on the evaluation dataset.

# %% [markdown]
# ##### A1 - Dataset Loading and Audit

# %%
raw_train_1 = mne.io.read_raw_gdf("../datasets/A01T.gdf")
raw_eval_1 = mne.io.read_raw_gdf("../datasets/A01E.gdf")

# %%
eog_chs = raw_train_1.ch_names[-3:]
raw_train_1.set_channel_types({ch: 'eog' for ch in eog_chs})

eog_chs_eval = raw_eval_1.ch_names[-3:]
raw_eval_1.set_channel_types({ch: 'eog' for ch in eog_chs_eval})

ch_names_2a = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
               'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
               'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right']
raw_train_1.rename_channels(dict(zip(raw_train_1.ch_names, ch_names_2a)))
raw_eval_1.rename_channels(dict(zip(raw_eval_1.ch_names, ch_names_2a)))

raw_train_1.set_montage(mne.channels.make_standard_montage('standard_1020'),
                         match_case=False, on_missing='ignore')
raw_eval_1.set_montage(mne.channels.make_standard_montage('standard_1020'),
                        match_case=False, on_missing='ignore')

# %%
raw_train_1

# %%
raw_eval_1

# %% [markdown]
# **Fill in:** shape, sampling rate, channel count/types — note anything
# unexpected (e.g. dropped channels, unusual sampling rate) from the raw
# object summaries printed above.

# %% [markdown]
# ##### A1b - Preprocessing: Notch filter, Bandpass filter, Common Average Reference
#
# Applied to the continuous raw signal, in this order, BEFORE epoching:
#   1. Notch filter — removes powerline interference (50 Hz mains + first
#      harmonic, since this data was recorded at Graz University of
#      Technology, Austria)
#   2. Bandpass filter — restricted to 1-40 Hz, wide enough to cover both
#      mu (8-12 Hz) and beta (13-30 Hz) bands for EDA/PSD exploration; a
#      narrower band is applied separately only for the CSP+LDA/SVM
#      classification pipeline later
#   3. Common Average Reference (CAR) — re-references every EEG channel to
#      the mean of all EEG channels, reducing reference-related noise
#
# A separate, pre-CAR copy (`raw_for_csd`) is branched off here for the
# Surface Laplacian (CSD) analysis further down, since CSD requires the
# data to have no custom reference already applied — CAR and CSD are two
# competing solutions to the same referencing problem, so only one of them
# would actually feed into the final classification pipeline; both are
# kept here in parallel purely for EDA comparison.

# %%
raw_train_1.load_data()

# 1. Notch filter — 50 Hz mains + first harmonic
raw_train_1.notch_filter(freqs=[50, 100], picks='eeg', verbose=False)

# 2. Bandpass filter — broad range for EDA; narrow later for classification
raw_train_1.filter(l_freq=1, h_freq=40, picks='eeg', verbose=False)

# branch off BEFORE CAR — CSD needs an unreferenced signal
raw_for_csd = raw_train_1.copy()

# 3. Common Average Reference — computed over EEG channels only, EOG excluded
raw_train_1.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

print("Preprocessing applied:", raw_train_1.info['highpass'], "-", raw_train_1.info['lowpass'], "Hz")
print("Reference:", raw_train_1.info['custom_ref_applied'])

# %% [markdown]
# **Fill in:** confirm the filter range and reference type printed above
# match what was intended; note why notch + bandpass + CAR were applied in
# this order (filtering before re-referencing avoids powerline artifact
# leaking into the average).

# %%
events, event_id = mne.events_from_annotations(raw_train_1, verbose=False)

marker_label = {
    1: "Rejected Trial",
    2: "Eye movments",
    3: "Idling EEG (eyes open)",
    4: "Idling EEG (eyes closed)",
    5: "Start of a new run",
    6: "Start of a trial",
    7: "left_hand",
    8: "right_hand",
    9: "feet",
    10: "tongue",
}

mi_event_id = {marker_label[v]: v for k, v in event_id.items()
               if marker_label[v] in ['left_hand', 'right_hand', 'feet', 'tongue']}

mi_epochs = mne.Epochs(raw_train_1, events, event_id=mi_event_id,
                        tmin=-0.5, tmax=5.25, baseline=None, preload=True)

# %% [markdown]
# **Fill in:** report dropped-epoch counts here (`mi_epochs.drop_log`) and
# whether any rejection criterion was applied — see the discussion on
# `IGNORED` vs genuine artifact-based drops before writing this up.

# %%
class_counts = pd.Series(mi_epochs.events[:, -1]).map(
    {v: k for k, v in mi_event_id.items()}
).value_counts()

fig, ax = plt.subplots(figsize=(5, 4))
class_counts.plot(kind='bar', ax=ax, color=sns.color_palette('colorblind'))
ax.set_ylabel("Number of trials")
ax.set_title("Class distribution — motor imagery trials")
plt.tight_layout()
plt.show()

print(class_counts)

# %% [markdown]
# **Fill in:** is the class distribution balanced? What does that imply for
# the choice of evaluation metrics (e.g. weighted F1) and train/test
# splitting strategy (stratified) later in the pipeline?

# %% [markdown]
# ##### A2 - EDA Visualization

# %% [markdown]
# ###### A2.1 - Mu-band ERD (%), C3 vs C4, left vs right hand
#
# ERD is computed as a per-trial PERCENT power change relative to a
# pre-stimulus baseline window, rather than comparing absolute PSD directly
# — absolute power is dominated by background 1/f spectrum and channel-level
# offsets unrelated to the task, so it does not show the contralateral
# effect on its own.

# %%
def compute_erd_percent(epochs, condition, channel, fmin=8, fmax=12,
                         baseline_window=(-0.5, 0), task_window=(0.5, 4.0)):
    ep = epochs[condition].copy()
    ch_idx = ep.ch_names.index(channel)

    base = ep.copy().crop(*baseline_window)
    task = ep.copy().crop(*task_window)

    psd_base = base.compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                 picks=[ch_idx], verbose=False).get_data().squeeze(axis=1)
    psd_task = task.compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                 picks=[ch_idx], verbose=False).get_data().squeeze(axis=1)

    base_power = psd_base.mean(axis=1)
    task_power = psd_task.mean(axis=1)
    erd_pct = 100 * (task_power - base_power) / base_power
    return erd_pct  # one value per trial

fig, ax = plt.subplots(figsize=(6, 4.5))
results = {}
for cls in ['left_hand', 'right_hand']:
    for channel in ['C3', 'C4']:
        erd = compute_erd_percent(mi_epochs, cls, channel)
        results[(cls, channel)] = erd

labels = ['C3', 'C4']
x = np.arange(len(labels))
width = 0.35

left_means = [results[('left_hand', ch)].mean() for ch in labels]
right_means = [results[('right_hand', ch)].mean() for ch in labels]

ax.bar(x - width / 2, left_means, width, label='left_hand')
ax.bar(x + width / 2, right_means, width, label='right_hand')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mu-band (8-12 Hz) ERD (%)")
ax.set_title("ERD relative to baseline — left vs right hand")
ax.legend()
plt.tight_layout()
plt.show()

print(pd.DataFrame({f"{cls}_mean_ERD%": [results[(cls, ch)].mean() for ch in labels]
                     for cls in ['left_hand', 'right_hand']}, index=labels))

# %% [markdown]
# **Fill in:** does right_hand show a larger (more negative) ERD at C3 than
# left_hand, and does left_hand show a larger ERD at C4? This is the
# per-channel version of the contralateral hypothesis — note whether it
# holds cleanly, partially, or not at this single-subject/single-session
# scale.

# %% [markdown]
# ###### A2.2 - Scalp topography of mu-band ERD%, CAR reference

# %%
epochs_topo = mi_epochs.copy().pick_types(eeg=True)
epochs_topo.set_montage(mne.channels.make_standard_montage('standard_1020'),
                         match_case=False, on_missing='ignore')

def compute_erd_percent_all_channels(epochs, condition, fmin=8, fmax=12,
                                      baseline_window=(-0.5, 0), task_window=(0.5, 4.0)):
    ep = epochs[condition].copy()
    base = ep.copy().crop(*baseline_window)
    task = ep.copy().crop(*task_window)

    psd_base = base.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False).get_data()
    psd_task = task.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False).get_data()

    base_power = psd_base.mean(axis=2).mean(axis=0)  # (n_channels,)
    task_power = psd_task.mean(axis=2).mean(axis=0)
    erd_pct = 100 * (task_power - base_power) / base_power
    return erd_pct

erd_left = compute_erd_percent_all_channels(epochs_topo, 'left_hand')
erd_right = compute_erd_percent_all_channels(epochs_topo, 'right_hand')

vlim = max(np.abs(erd_left).max(), np.abs(erd_right).max())

fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1, 0.08]})

mne.viz.plot_topomap(erd_left, epochs_topo.info, axes=axes[0], show=False,
                      cmap='RdBu_r', vlim=(-vlim, vlim))
axes[0].set_title('Left hand imagery')

mne.viz.plot_topomap(erd_right, epochs_topo.info, axes=axes[1], show=False,
                      cmap='RdBu_r', vlim=(-vlim, vlim))
axes[1].set_title('Right hand imagery')

norm = plt.Normalize(vmin=-vlim, vmax=vlim)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
fig.colorbar(sm, cax=axes[2], label='ERD (%)')

fig.suptitle('Mu-band (8-12 Hz) ERD topography — CAR reference: blue = power decrease')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Fill in:** note the dominant patch in each topomap. If a strong
# posterior/parietal effect is visible in BOTH classes, that is consistent
# with a shared, non-lateralized process (e.g. general visual/attentional
# alpha blocking to the cue) rather than the sensorimotor effect of
# interest — see the difference map below for the part that is actually
# task-specific.

# %% [markdown]
# ###### A2.3 - ERD difference map (left − right), isolating the lateralized component
#
# Subtracting right_hand ERD% from left_hand ERD% cancels anything common
# to both conditions and isolates the part that differs by imagined hand —
# i.e. the contralateral signature we are actually looking for.

# %%
erd_diff = erd_left - erd_right  # positive => more suppression in right_hand
vlim_diff = np.abs(erd_diff).max()

fig, ax = plt.subplots(figsize=(5, 4.5))
im, _ = mne.viz.plot_topomap(erd_diff, epochs_topo.info, axes=ax, show=False,
                              cmap='RdBu_r', vlim=(-vlim_diff, vlim_diff))
ax.set_title('ERD difference: left_hand − right_hand')
plt.colorbar(im, ax=ax, label='ERD% difference')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Fill in:** a positive (red) patch over C3 and a negative (blue) patch
# over C4 in this difference map — or vice versa — is the actual signature
# of contralateral lateralization. Note whether this pattern is visible now
# that the shared posterior effect has been subtracted out.

# %% [markdown]
# ###### A2.4 - Scalp topography of mu-band ERD%, Surface Laplacian (CSD) reference
#
# The surface Laplacian is applied to `raw_for_csd` — the pre-CAR branch
# saved during preprocessing — since CSD requires the data to carry no
# custom reference. CSD emphasises focal, local sources and is less prone
# to being dominated by a strong signal at sparsely-covered sites (e.g. the
# posterior effect seen with CAR), so it may sharpen the C3/C4 contralateral
# pattern if it is genuinely present. Note that CSD units (µV/cm²) differ
# from CAR (µV), so only the spatial pattern should be compared between the
# two, not the absolute ERD% scale.

# %%
raw_csd = mne.preprocessing.compute_current_source_density(raw_for_csd, verbose=False)

epochs_csd = mne.Epochs(raw_csd, events, event_id=mi_event_id,
                         tmin=-0.5, tmax=5.25, baseline=None, preload=True, verbose=False)
epochs_csd.pick_types(eeg=True)

erd_left_csd = compute_erd_percent_all_channels(epochs_csd, 'left_hand')
erd_right_csd = compute_erd_percent_all_channels(epochs_csd, 'right_hand')

vlim_csd = max(np.abs(erd_left_csd).max(), np.abs(erd_right_csd).max())

fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1, 0.08]})

mne.viz.plot_topomap(erd_left_csd, epochs_csd.info, axes=axes[0], show=False,
                      cmap='RdBu_r', vlim=(-vlim_csd, vlim_csd))
axes[0].set_title('Left hand imagery')

mne.viz.plot_topomap(erd_right_csd, epochs_csd.info, axes=axes[1], show=False,
                      cmap='RdBu_r', vlim=(-vlim_csd, vlim_csd))
axes[1].set_title('Right hand imagery')

norm = plt.Normalize(vmin=-vlim_csd, vmax=vlim_csd)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
fig.colorbar(sm, cax=axes[2], label='ERD (%)')

fig.suptitle('Mu-band ERD topography — Surface Laplacian (CSD) reference')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Fill in:** compare this to the CAR version (A2.2) — is the posterior
# effect reduced, and is there now a clearer focal patch over C4
# (left_hand) and C3 (right_hand)? State which referencing scheme (CAR or
# CSD) will actually be carried forward into the CSP/classification
# pipeline, and why.