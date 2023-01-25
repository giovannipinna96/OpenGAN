import matplotlib.pylab as plt
from utils.eval_funcs import *


def plot_losses(G_losses, D_losses, modelFlag, xlabel='Iterations', ylabel='Loss'):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('learningCurves_{}.png'.format(modelFlag), bbox_inches='tight',transparent=True)
    plt.close()


def plot_roc_curve(outputs_open, outputs_close, modelFlag, xlabel='False Positive Rate', ylabel='True Positive Rate'):
    roc_score, roc_to_plot = evaluate_openset(outputs_open, outputs_close)
    plt.plot(roc_to_plot['fp'], roc_to_plot['tp'])
    plt.grid('on')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('ROC score {:.5f}'.format(roc_score))
    plt.savefig('roc_{}.png'.format(modelFlag), bbox_inches='tight',transparent=True)
    plt.close()

def plot_hist(outputs_open, outputs_close, modelFlag, label_hist_open="punzoni extra", label_hist_close='punzoni closeset'):
    plt.hist(outputs_open, label=label_hist_open, density=True, alpha=0.5)
    plt.hist(outputs_close, label=label_hist_close, density=True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('Overlapping')
    plt.savefig('hist_{}.png'.format(modelFlag), bbox_inches='tight',transparent=True)
    plt.close()