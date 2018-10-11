import matplotlib.pyplot as plt

def plot_history(hist):
    height = (len(hist.history)+1)//4+1
    figs, axes = plt.subplots(height, 2, figsize=(15,5*height))
    keys = list(filter(lambda x: x.startswith('val'),history.history.keys()))
    for i,key in enumerate(keys):
        axes[i//2,i%2].plot(hist.epoch, hist.history[key[4:]], label="Train "+key[4:])
        axes[i//2,i%2].plot(hist.epoch, hist.history[key], label="Validation "+key)
        axes[i//2,i%2].legend()
    i += 1
    axes[i//2,i%2].plot(hist.epoch, hist.history['lr'], label="learning rate")
    axes[i//2,i%2].legend()
