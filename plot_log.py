import json
import matplotlib.pyplot as plt

experiment_folder = './output/DABIDOL_Base_YTVIS19_R50'
experiment_folder_IDOL = './output/IDOL_YTVIS19_R50'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
experiment_metrics_IDOL = load_json_arr(experiment_folder_IDOL + '/metrics.json')

# print(experiment_metrics)

for x in experiment_metrics:
    print(x['total_loss'])
    break

plt.plot(
    [x['iteration'] for x in experiment_metrics],
    [x['total_loss'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics_IDOL],
    [x['total_loss'] for x in experiment_metrics_IDOL])
plt.legend(['total_loss_DAB', 'total_loss_IDOL'], loc='upper left')
plt.show()