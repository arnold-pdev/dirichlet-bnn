using PyCall

py"""
# %% Extract and prepare data for dirichlet regression

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# %%
ox_wt_names = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]

def create_masks(wav):
    uv_mask = np.logical_and(wav > uv_range[0], wav < uv_range[1])
    vio_mask = np.logical_and(wav > vio_range[0], wav < vio_range[1])
    vnir_mask = np.logical_and(wav > vnir_range[0], wav < vnir_range[1])
    return uv_mask, vio_mask, vnir_mask

def rescale_libs(libs, uv_mask, vio_mask, vnir_mask):
    # %% Per-spectrometer rescaling by sum of intensities; suggested by Patrick, though other options are possible. Also ensures all values are non-negative for NMF.

    libs[libs < 0] = 0
    libs[:, uv_mask] = libs[:, uv_mask] / np.sum(libs[:, uv_mask], axis=1, keepdims=True)
    libs[:, vio_mask] = libs[:, vio_mask] / np.sum(libs[:, vio_mask], axis=1, keepdims=True)
    libs[:, vnir_mask] = libs[:, vnir_mask] / np.sum(libs[:, vnir_mask], axis=1, keepdims=True)
    return 

# %% Load composition data
comp = pd.read_csv('chemcam-data/ccam-libs-calibration-moc-v3-for-pds.csv')
# Drop NA rows
comp = comp.loc[np.logical_not(comp.isnull().all(1))]
comp = comp.loc[:, ['Target', 'MOC total'] + ox_wt_names]
comp[ox_wt_names].apply(pd.to_numeric, errors='coerce')

# %% Load Earth calibration data
# Two-row header... get columns for LIBS data first
earth_cal_row1 = pd.read_csv('chemcam-data/Supplement_MnO_Cal_Input_outliers_wvl.csv', nrows=1)
wav_col_ind = [i for i, c in enumerate(list(earth_cal_row1.columns)) if "wvl" in c]
# Load full data frame with second row as header
earth_cal = pd.read_csv('chemcam-data/Supplement_MnO_Cal_Input_outliers_wvl.csv', header=1)
# Extract wavelength values from column names
earth_cal_wav_str = np.array([c for i, c in enumerate(list(earth_cal.columns)) if i in wav_col_ind])
earth_cal_wav = np.array(earth_cal_wav_str).astype("float64")

# %% Find intersection of targets
comp_trg = np.unique(comp["Target"].to_numpy(copy=True).astype("str"))
earth_cal_trg = np.unique(earth_cal["Target"].to_numpy(copy=True).astype("str"))
print('LIBS unique targets %d ' % len(earth_cal_trg))
intersect_trg = list(set(comp_trg).intersection(set(earth_cal_trg)))
print('Unique comp_trg N=%d, earth_cal_trg N=%d, intersection N=%d' % (len(comp_trg), len(earth_cal_trg), len(intersect_trg)))

# %% Subset to targets
comp = comp.loc[comp["Target"].isin(intersect_trg), :]
earth_cal = earth_cal.loc[earth_cal["Target"].isin(intersect_trg), :]

# %% Get relevant columns of earth cal
wav_col_ind = [i for i, c in enumerate(list(earth_cal_row1.columns)) if "wvl" in c]
wav_col = [c for i, c in enumerate(list(earth_cal.columns)) if i in wav_col_ind]
earth_cal = earth_cal.loc[:, ["Target"] + wav_col]

# %% Mergecomp and data
earth_cal_comp = comp.merge(earth_cal, how='outer', on="Target")

# %% Get intensity data for PCA
x = earth_cal_comp.loc[:, wav_col].to_numpy()

# %% Preprocess
uv_mask, vio_mask, vnir_mask = create_masks(earth_cal_wav)
rescale_libs(x, uv_mask, vio_mask, vnir_mask)
mask = np.logical_or(uv_mask, np.logical_or(vio_mask, vnir_mask))
x = x[:, mask]

# %% Do PCA
pc = PCA(n_components=50, whiten=True)
pc = pc.fit(x)

plt.figure()
plt.plot(np.arange(50), np.cumsum(pc.explained_variance_ratio_))
plt.axhline(0.99)
plt.show()

print('Explained variance ratio: %s' % pc.explained_variance_ratio_)
pc = PCA(n_components=25, whiten=True)
pc_x = pc.fit_transform(x)
print('Explained variance ratio: %s' % pc.explained_variance_ratio_)

# %% Create data frame of PCA
df = {'PC%d' % (i+1): pc_x[:, i] for i in range(25)}
df['Target'] = earth_cal_comp['Target']
df = pd.DataFrame(df)
print(df.head())

# %% Merge with comp
save_df = earth_cal_comp.loc[:, ['Target', 'MOC total'] + ox_wt_names]
for i in range(25):
    save_df['PC%d' % (i+1)] = np.nan
    print('PC%d' % (i+1))
for t in np.unique(save_df['Target']):
    for i in range(25):
        save_df.loc[save_df['Target']==t, 'PC%d' % (i+1)] = df.loc[df['Target']==t, 'PC%d' % (i+1)]
        print('Target %s, PC%d, N=%d' % (t, i+1, np.sum(save_df['Target']==t)))
# %% Write to file
save_df.to_csv('chemcam-data/ChemCam_PCA_composition.csv', na_rep=0.0, index=False)

# %% Save PCA components
np.save('chemcam-data/ChemCam_PCA_components.npy', pc.components_)

# %%
"""