import pickle
import matplotlib.pyplot as plt

''' takes in 3D array of sequential [x,y] '''
def plot_deviation(trajectories, model, deflation_pattern, centerline, roadleft, roadright, savefile="trajectories"):
    x, y = [], []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    x, y = [], []
    for point in roadleft:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    x, y = [], []
    for point in roadright:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-", label="Road")
    for i,t in enumerate(trajectories):
        x,y = [],[]
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, label="Run {}".format(i), alpha=0.75)
    x.sort()
    y.sort()
    min_x, max_x = x[0], x[-1]
    min_y, max_y = y[0], y[-1]
    plt.xlim(min_x - 5, max_x + 5)
    plt.ylim(min_y-5, max_y + 5)
    plt.title(f'Trajectories with {model} \n{savefile}')
    plt.legend(loc=2, prop={'size': 6})
    plt.draw()
    print(f"Saving image to {deflation_pattern}/{savefile}.jpg")
    plt.savefig(f"{savefile}.jpg")
    # plt.show()
    # plt.pause(0.1)

# filename = "F:\DAVE2v3-81x144-99samples-1000epoch-5108267-4_10-12_26-DGB5XQ\industrial-7982-tracktopo-10runs/summary-model-DAVE2v3-randomblurnoise-81x144-lr1e4-1000epoch-64batch-lossMSE-99Ksamples.pickle"
# filename = "F:\DAVE2v3-108x192-82samples-1000epoch-5116933-4_10-17_9-2RIWIM\industrial-7982-tracktopo-10runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-1000epoch-64batch-lossMSE-82Ksamples.pt.pickle"
filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\hirochi_raceway-9039-Rturntopo-5runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_vqvae_data_RLRturnfisheye.pth.pickle"
filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\hirochi_raceway-9039-Rturntopo-5runs\\summary-model-DAVE2v3VAE-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_vqvae_data_RLRturnfisheye.pth.pickle"
#filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\automation_test_track-8185-straighttopo-5runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_vqvae_data_RLstraightfisheye.pth.pickle"
#filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\automation_test_track-8185-straighttopo-5runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_vqvae_data_RLstraightfisheye.pth.pickle"
#filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\west_coast_usa-12930-Lturntopo-5runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_vqvae_data_RLLturnfisheye.pth.pickle"
# filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\west_coast_usa-12930-Lturntopo-5runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_None.pickle"
#filename = "F:\\DAVE2-Keras\\DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63\\automation_test_track-8185-straighttopo-5runs\\summary-model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-best152.pt_noVQVAE.pickle"
with open(filename, 'rb') as f:
    x = pickle.load(f)
    savefile = filename.replace(".pth.pickle", "")
    plot_deviation(x["trajectories"], "DAVE2V3 ", ".", x["centerline_interpolated"], x["roadleft"], x["roadright"], savefile=savefile)
    print(f"OUT OF {len(x['trajectories'])} RUNS:"
          f"\n\tAvg. distance: {(sum(x['dists_travelled'])/len(x['dists_travelled'])):.4f}"
          f"\n\tAvg. deviation: {(sum(x['dists_from_centerline']) / len(x['dists_from_centerline'])):.4f}"
          # f"\n\t{distances=}"
          # f"\n\t{deviations:}"
          )