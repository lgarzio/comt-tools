import pickle5 as pickle

plt_labels = {'wrf1': ['k-kl', 'tab:blue'], 'wrf3': [r'k-$\epsilon$', 'tab:orange'],
              'wrf4': [r'k-$\omega$', 'tab:olive'], 'wrf5': ['HyCOM', 'tab:pink'], 'wrf6': ['MUR SST', 'tab:purple'],
              'roms5': ['HyCOM', 'tab:pink'], 'roms6': ['MUR SST', 'tab:purple']}

with open('plt_labels.pickle', 'wb') as handle:
    pickle.dump(plt_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)