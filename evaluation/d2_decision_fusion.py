import pandas as pd
import numpy as np
from evaluation.a_performance_metrics import performance_measures, non_binary_metrics, binary_metrics
import matplotlib.pyplot as plt
import os
from evaluation.utilities import plot_pred_trues

# Find the predictions of unimodal models
test_on = 'dev'
predictions_folder = 'C:/Users/mjkoe/Thesis/results/predictions_best_settings/'

if test_on == 'dev':
    unimodal_pred_w2v = os.path.join(predictions_folder, 'pdem_wav2vec_symp_60_var_std_quantile_dev.csv')
    unimodal_pred_text = os.path.join(predictions_folder, 'sbert_all-MiniLM-L12-v2_symp_50_mean_median_dev.csv')
    unimodal_pred_vad = os.path.join(predictions_folder, 'pdem_vad_symp_4_mean_max_dev.csv')
else:
    unimodal_pred_w2v = os.path.join(predictions_folder, 'pdem_wav2vec_symp_60_var_std_quantile_test.csv')
    unimodal_pred_text = os.path.join(predictions_folder, 'sbert_all-MiniLM-L12-v2_symp_50_mean_median_test.csv')
    unimodal_pred_vad = os.path.join(predictions_folder, 'pdem_vad_symp_4_mean_max_test.csv')

df_w2v_pred = pd.read_csv(unimodal_pred_w2v, index_col=0)
df_text_pred = pd.read_csv(unimodal_pred_text, index_col=0)
df_vad_pred = pd.read_csv(unimodal_pred_vad, index_col=0)

true_severity = df_w2v_pred['true_severity']

w2v_pred_severity = df_w2v_pred['pred_severity']
vad_pred_severity = df_vad_pred['pred_severity']
text_pred_severity = df_text_pred['pred_severity']

# """ Equal weighting of audio and text modalities (severity fusion)"""
# w_audio = 0.5
# w_text = 1 - w_audio
# bimodal_pred_severity = (w_audio * w2v_pred_severity + w_text * text_pred_severity).astype(int)
#
# df_bimodal_predictions = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_w2v_pred['true_binary'],
#                                     'gender': df_w2v_pred['gender'], 'pred_severity': bimodal_pred_severity})
# print('\nEqual weighting (severity fusion) sbert + pdem')
# performance_measures(df_bimodal_predictions)  # print performance
#
#
# """ Max of audio and text modalities (severity fusion)"""
# max_sev = []
# for i in range(len(w2v_pred_severity)):
#     max_sev.append(max([w2v_pred_severity.values[i], text_pred_severity.values[i]]))
# bimodal_pred_severity = max_sev
#
# df_bimodal_predictions = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_w2v_pred['true_binary'],
#                                     'gender': df_w2v_pred['gender'], 'pred_severity': bimodal_pred_severity})
#
#
# print('\nMax (severity decision fusion) sbert + pdem')
# performance_measures(df_bimodal_predictions)  # print performance
#
#

""" Check for different weightings of the two modalities to improve performance"""
if test_on == 'dev':
    perf_lists = [[] for _ in range(9)]

    w_list = np.arange(0.0, 1, 0.01)
    for w in w_list:
        w_audio = w
        w_text = 1 - w_audio

        bimodal_pred_severity = (w_audio * df_w2v_pred['pred_severity']
                                 + w_text * df_text_pred['pred_severity'])
        bimodal_pred_severity_g0 = (w_audio * df_w2v_pred.query('gender == 0')['pred_severity']
                                    + w_text * df_text_pred.query('gender == 0')['pred_severity'])
        bimodal_pred_severity_g1 = (w_audio * df_w2v_pred.query('gender == 1')['pred_severity']
                                    + w_text * df_text_pred.query('gender == 1')['pred_severity'])

        rms, cc, ma = non_binary_metrics(true_severity, bimodal_pred_severity)
        rms_g0, cc_g0, ma_g0 = non_binary_metrics(df_w2v_pred.query('gender == 0')['true_severity'],
                                                  bimodal_pred_severity_g0)
        rms_g1, cc_g1, ma_g1 = non_binary_metrics(df_w2v_pred.query('gender == 1')['true_severity'],
                                                  bimodal_pred_severity_g1)

        scores = [rms, rms_g0, rms_g1, cc, cc_g0, cc_g1, ma, ma_g0, ma_g1]

        for i, perf_list in enumerate(perf_lists):
            perf_list.append(scores[i])

    color_list = ['purple', 'r', 'b']
    labels = ['Overall', 'Female', 'Male']
    i_perf_list = [0, 3, 6]

    # TODO: get optimal weights from development set, then test on test_set

    # Create subplots of gender-specific weights
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey='col')  # 1 row, 3 columns

    for i in [0, 1, 2]:
        for j, score in enumerate(['RMSE', 'CCC', 'MAE']):
            axes[j].set(xlabel='Audio weight_kelm', ylabel=score)
            axes[j].axvline(x=0.5, lw=1, c='black', linestyle='dotted')

            y = perf_lists[i+i_perf_list[j]]
            axes[j].plot(w_list, y, lw=1, c=color_list[i], label=labels[i])
            if score == 'CCC':
                axes[j].scatter(x=w_list[y.index(max(y))], y=max(y), s=20, c=color_list[i], marker='p')
            else:
                axes[j].scatter(x=w_list[y.index(min(y))], y=min(y), s=20, c=color_list[i], marker='p')

    axes[0].legend()
    plt.tight_layout()
    plt.show()
#
#     # Print performances and plot predictions of optimized-decision level fusion
#     best_g0 = w_list[perf_lists[1].index(min(perf_lists[1]))]
#     best_g1 = w_list[perf_lists[2].index(min(perf_lists[2]))]
#
# else:
#     best_g0 = 0
#     best_g1 = 0.99
#
# w_text_g0 = 1 - best_g0
# w_text_g1 = 1 - best_g1
#
# bimodal_pred_severity_best_g0 = (best_g0 * df_w2v_pred['pred_severity'] + w_text_g0 * df_text_pred['pred_severity']).astype(int)
# bimodal_pred_severity_best_g1 = (best_g1 * df_w2v_pred['pred_severity'] + w_text_g1 * df_text_pred['pred_severity']).astype(int)
#
# optimized_for_g0 = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_w2v_pred['true_binary'],
#                                  'gender': df_w2v_pred['gender'], 'pred_severity': bimodal_pred_severity_best_g0})
# optimized_for_g1 = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_w2v_pred['true_binary'],
#                                  'gender': df_w2v_pred['gender'], 'pred_severity': bimodal_pred_severity_best_g1})
#
# df_g0 = optimized_for_g0.query('gender == 0')
# df_g1 = optimized_for_g1.query('gender == 1')
#
# df_optimal_fusion_gender = pd.concat([df_g0, df_g1], ignore_index=True)
#
# # Show performance
# print(f'\nOptimal weighting audio f: {best_g0}, audio m: {best_g1} (severity decision fusion)')
# performance_measures(df_optimal_fusion_gender)
# # plot_pred_trues(df_optimal_fusion_gender)


#
# """ Equal weighting of vad and text modalities (severity fusion)"""
# w_audio = 0.5
# w_text = 1 - w_audio
# bimodal_pred_severity = (w_audio * vad_pred_severity + w_text * text_pred_severity).astype(int)
#
# df_bimodal_predictions = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_vad_pred['true_binary'],
#                                     'gender': df_vad_pred['gender'], 'pred_severity': bimodal_pred_severity})
# print('\nEqual weighting (severity fusion) sbert + vad')
# performance_measures(df_bimodal_predictions)  # print performance
#
#
# """ Max of audio and text modalities (severity fusion)"""
# max_sev = []
# for i in range(len(vad_pred_severity)):
#     max_sev.append(max([vad_pred_severity.values[i], text_pred_severity.values[i]]))
# bimodal_pred_severity = max_sev
#
# df_bimodal_predictions = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_vad_pred['true_binary'],
#                                     'gender': df_vad_pred['gender'], 'pred_severity': bimodal_pred_severity})
#
#
# print('\nMax (severity decision fusion) sbert + vad')
# performance_measures(df_bimodal_predictions)  # print performance
#
# """ Hybrid fusion with gender specific weights"""
# best_g0 = 0
# best_g1 = 0.8
#
# w_text_g0 = 1 - best_g0
# w_text_g1 = 1 - best_g1
#
# test_on = 'dev'
# predictions_folder_best_g0 = 'C:/Users/mjkoe/Thesis/results/predictions_best_settings/vad_weighted/'
# predictions_folder_best_g1 = 'C:/Users/mjkoe/Thesis/results/predictions_best_settings/'
#
#
# if test_on == 'dev':
#     best_model_for_g0 = os.path.join(predictions_folder_best_g0, 'sbert_miniLM_weighted_symp_sbert_arousal_abs_dev.csv')
#     df_best_model_for_g0 = pd.read_csv(best_model_for_g0, index_col=0)
#     best_model_for_g1 = os.path.join(predictions_folder_best_g1, 'pdem_wav2vec_symp_60_var_std_quantile_dev.csv')
#     df_best_model_for_g1 = pd.read_csv(best_model_for_g1, index_col=0)
# else:
#     best_model_for_g0 = os.path.join(predictions_folder_best_g0, 'sbert_miniLM_weighted_symp_sbert_arousal_abs_test.csv')
#     df_best_model_for_g0 = pd.read_csv(best_model_for_g0, index_col=0)
#     best_model_for_g1 = os.path.join(predictions_folder_best_g1, 'pdem_wav2vec_symp_60_var_std_quantile_test.csv')
#     df_best_model_for_g1 = pd.read_csv(best_model_for_g1, index_col=0)
#
# df_g0 = df_best_model_for_g0.query('gender == 0')
# df_g1 = df_best_model_for_g1.query('gender == 1')
#
# df_optim = pd.concat([df_g0, df_g1], ignore_index=True)
# plot_pred_trues(df_optim)
# # Show performance
# performance_measures(df_optim)
# plot_pred_trues(df_optimal_fusion_gender)
#
# bimodal_pred_severity_best_g0 = (best_g0 * df_best_model_for_g0['pred_severity'] +
#                                  w_text_g0 * df_best_model_for_g1['pred_severity']).astype(int)
# bimodal_pred_severity_best_g1 = (best_g1 * df_best_model_for_g0['pred_severity'] +
#                                  w_text_g1 * df_best_model_for_g1['pred_severity']).astype(int)
#
# optimized_for_g0 = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_best_model_for_g0['true_binary'],
#                                  'gender': df_best_model_for_g0['gender'], 'pred_severity': bimodal_pred_severity_best_g0})
# optimized_for_g1 = pd.DataFrame({'true_severity': true_severity, 'true_binary': df_best_model_for_g1['true_binary'],
#                                  'gender': df_best_model_for_g1['gender'], 'pred_severity': bimodal_pred_severity_best_g1})
#
# df_g0 = optimized_for_g0.query('gender == 0')
# df_g1 = optimized_for_g1.query('gender == 1')
#
# df_optimal_fusion_gender = pd.concat([df_g0, df_g1], ignore_index=True)
#
# # Show performance
# print(f'\nOptimal weighting audio f: {best_g0}, audio m: {best_g1} (severity decision fusion)')
# performance_measures(df_optimal_fusion_gender)
# # plot_pred_trues(df_optimal_fusion_gender)