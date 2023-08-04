import json
import os
import pandas as pd
from glob import glob
from PIL import Image
import statistics

from metrics_util import precision_recall, metrics_table, metrics_col, metrics_row, check_table, \
    check_column, check_row
from global_variables import LOCAL_DATA_DIR, TEST_S3_BUCKET, TEST_S3_PATH, BEST_RESULT_S3_PATH
from lambda_utils import call_email_lambda, invoke_lambda
from utils import s3_cp, s3_sync, get_best_result


def add(cum_TP, cum_FP, cum_FN, TP, FP, FN):
    return cum_TP + TP, cum_FP + FP, cum_FN + FN


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def get_score(df, real_path, pred_path, thresholds):
    """
        Getting table, column and row score for prediction. Returns dictionary of results with scores.
    :param df: CSV with test set details.
    :param real_path: Path to real labels.
    :param pred_path: Path to predicted labels.
    :param thresholds: IOU threshold
    :return: Dictionary of results with scores
    """
    result = dict()
    for thresh_iou in thresholds:
        cum_TP_col, cum_FP_col, cum_FN_col, cum_TP_row, cum_FP_row, cum_FN_row = 0, 0, 0, 0, 0, 0
        table_score_ls = []
        for filename, page_no in zip(df['file_name'], df['page_number_index']):
            page_no -= 1
            doc_id = filename[:filename.rfind('.')]
            filename = filename[:filename.rfind('.')] + '.json'

            if check_table(real_path, pred_path, filename):
                table_score = metrics_table(real_path, pred_path, filename)
                table_score_ls.append(table_score)

            if check_column(real_path, pred_path, filename):
                TP, FP, FN = metrics_col(real_path, pred_path, filename, doc_id, thresh_iou)
                cum_TP_col, cum_FP_col, cum_FN_col = add(cum_TP_col, cum_FP_col, cum_FN_col, TP, FP, FN)

            if check_row(real_path, pred_path, filename):
                TP, FP, FN = metrics_row(real_path, pred_path, filename, doc_id, thresh_iou, page_no)
                cum_TP_row, cum_FP_row, cum_FN_row = add(cum_TP_row, cum_FP_row, cum_FN_row, TP, FP, FN)

        thresh_key = str(thresh_iou)
        result[thresh_key] = dict()

        if check_table(real_path, pred_path, filename):
            avg_table_score = sum(table_score_ls) / len(table_score_ls)
            result[thresh_key]['Average Table Score'] = avg_table_score

        if check_column(real_path, pred_path, filename):
            precision_col, recall_col = precision_recall(cum_TP_col, cum_FP_col, cum_FN_col)
            result[thresh_key]['Column Seprators Precision'] = precision_col
            result[thresh_key]['Column Seprators Recall'] = recall_col
            result[thresh_key]['Column Seprators F1 Score'] = f1_score(precision_col, recall_col)

        if check_row(real_path, pred_path, filename):
            precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
            result[thresh_key]['Row Seprators Precision'] = precision_row
            result[thresh_key]['Row Seprators Recall'] = recall_row
            result[thresh_key]['Row Seprators F1 Score'] = f1_score(precision_row, recall_row)

    return result


def get_bucket_analysis(df_org, real_path, pred_path, thresholds):
    """
        Function for bucket analysis i.e. get score on each bucket of test data.
    :param df_org: CSV with test set details.
    :param real_path: Path to real labels.
    :param pred_path: Path to predicted labels.
    :param thresholds: IOU threshold
    :return: Dictionary of results with scores and email output.
    """
    result = dict()
    output = 'Bucket Analysis: <br>'
    col_ls = ['format']
    for col in col_ls:
        result[col] = dict()
        for bucket_type in df_org[col].unique():
            if pd.isna(bucket_type):
                continue
            result[col][bucket_type] = dict()
            df = df_org[df_org[col] == bucket_type]
            output += '******************************************************************************************<br>'
            output += f'Column = {col} <br>'
            output += f'Bucket Type = {bucket_type} <br>'
            for thresh_iou in thresholds:
                result[col][bucket_type][thresh_iou] = dict()
                cum_TP_col, cum_FP_col, cum_FN_col, cum_TP_row, cum_FP_row, cum_FN_row = 0, 0, 0, 0, 0, 0
                table_score_ls = []
                for filename, page_no in zip(df['file_name'], df['page_number_index']):
                    page_no -= 1
                    doc_id = filename[:filename.rfind('.')]
                    filename = filename[:filename.rfind('.')] + '.json'
                    if check_table(real_path, pred_path, filename):
                        table_score = metrics_table(real_path, pred_path, filename)
                        table_score_ls.append(table_score)

                    if check_column(real_path, pred_path, filename):
                        TP, FP, FN = metrics_col(real_path, pred_path, filename, doc_id, thresh_iou)
                        cum_TP_col += TP
                        cum_FP_col += FP
                        cum_FN_col += FN

                    if check_row(real_path, pred_path, filename):
                        TP, FP, FN = metrics_row(real_path, pred_path, filename, doc_id, thresh_iou)
                        cum_TP_row += TP
                        cum_FP_row += FP
                        cum_FN_row += FN

                output += f'For Thresh IOU = {thresh_iou} <br>'

                if check_table(real_path, pred_path, filename):
                    avg_table_score = sum(table_score_ls) / len(table_score_ls)
                    output += f'Average Table Score = {avg_table_score:.4f} <br>'
                    result[col][bucket_type][thresh_iou]['Average Table Score'] = avg_table_score

                if check_column(real_path, pred_path, filename):
                    precision_col, recall_col = precision_recall(cum_TP_col, cum_FP_col, cum_FN_col)
                    output += f'For Column Seprators <br>'
                    result[col][bucket_type][thresh_iou]['Column Seprators'] = dict()
                    output += f'Precision = {precision_col:.4f} <br>'
                    output += f'Recall = {recall_col:.4f} <br>'
                    output += f'F1 Score = {f1_score(precision_col, recall_col):.4f} <br>'
                    result[col][bucket_type][thresh_iou]['Column Seprators']['TP'] = cum_TP_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['FP'] = cum_FP_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['FN'] = cum_FN_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['Precision'] = precision_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['Recall'] = recall_col
                    result[col][bucket_type][thresh_iou]['Column Seprators']['F1 Score'] = f1_score(precision_col,
                                                                                                    recall_col)

                if check_row(real_path, pred_path, filename):
                    precision_row, recall_row = precision_recall(cum_TP_row, cum_FP_row, cum_FN_row)
                    output += f'For Row Seprators <br>'
                    result[col][bucket_type][thresh_iou]['Row Seprators'] = dict()
                    output += f'Precision = {precision_row:.4f} <br>'
                    output += f'Recall = {recall_row:.4f} <br>'
                    output += f'F1 Score = {f1_score(precision_row, recall_row):.4f} <br>'
                    result[col][bucket_type][thresh_iou]['Row Seprators']['TP'] = cum_TP_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['FP'] = cum_FP_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['FN'] = cum_FN_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['Precision'] = precision_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['Recall'] = recall_row
                    result[col][bucket_type][thresh_iou]['Row Seprators']['F1 Score'] = f1_score(precision_row,
                                                                                                 recall_row)
                output += '<br>'

    return result, output


def totals_box_removal(xywh):
    tops = [dims[1] - dims[3] / 2 for dims in xywh]
    end = min([dims[1] + dims[3] / 2 for dims in xywh])
    to_remove = []
    for i in range(len(tops)):
        if tops[i] > end:
            to_remove += i,
    if not to_remove:
        return xywh
    else:
        for i in reversed(range(len(to_remove))):
            xywh.pop(to_remove[i])
        return xywh


def get_col_seps(xywh):
    if not xywh:
        return ''
    threshold = 0.025
    xywh = totals_box_removal(xywh)
    starts_ends = [((info[0] - info[2] / 2), (info[0] + info[2] / 2)) for info in xywh]
    col_seps_ratio = []
    for i in range(len(xywh) - 1):
        cs1, cs2 = starts_ends[i][1], starts_ends[i + 1][0]
        if cs2 - cs1 < threshold:
            col_seps_ratio += (cs1 + cs2) / 2,
        else:
            col_seps_ratio += cs1,
            col_seps_ratio += cs2,
    col_seps = [str(sep) for sep in col_seps_ratio]
    seps_str = ','.join(col_seps)
    return seps_str


def get_table_dims(xywh):
    if not xywh:
        return ''
    xywh.sort(key=lambda x: x[0])
    l = xywh[0][0] - xywh[0][2] / 2
    r = xywh[-1][0] + xywh[-1][2] / 2
    tops = [dims[1] - dims[3] / 2 for dims in xywh]
    # t = sum(tops) / len(tops)
    t = statistics.median(tops)
    botts = [dims[1] + dims[3] / 2 for dims in xywh]
    # b = sum(botts) / len(botts)
    b = statistics.median(botts)
    tab_dims = [l, t, r, b]
    tab_dims = list(map(str, tab_dims))
    dims_str = ','.join(tab_dims)
    return dims_str


def column_model_inference(model, image_path):
    image = Image.open(image_path)
    result = model(image, verbose=True)
    box_xywhn_ls = []
    for res in result:
        for box in res.boxes:
            box_xywhn_ls.append(box.xywhn.tolist()[0])
    box_xywhn_ls = sorted(box_xywhn_ls, key=lambda x: x[0])
    col_seps = get_col_seps(box_xywhn_ls)
    table_dims = get_table_dims(box_xywhn_ls)
    output = {'column_separators': col_seps, 'table_coordinates': table_dims}
    return output


def get_model_output_multiprocessing(model_col):
    image_paths = glob(f'{LOCAL_DATA_DIR}/images/*.png')
    for image_path in image_paths:
        print(image_path)
        image_name = image_path[image_path.rfind('/')+1:]
        label_name = image_name[:-3] + 'json'
        output = column_model_inference(model_col, image_path)
        if output['column_separators'] is None:
            output['column_separators'] = ''
            output['table_coordinates'] = ''
        with open(f'{LOCAL_DATA_DIR}/model_outputs/{label_name}', 'w') as f:
            json.dump(output, f)


def get_mail_body(model_result):
    """
        Convert score result to email output format.
    :param best_result: Best run result.
    :param model_result: Current model result.
    :return: Email output.
    """
    output = ''
    for thresh_iou in model_result.keys():
        output += f'For Thresh IOU = {thresh_iou} <br>'
        for score in model_result[thresh_iou].keys():
            output += f'{score} for model={model_result[thresh_iou][score]:.4f} <br>'
        output += '<br>'
    return output


def download_test_data():
    s3_sync(f's3://{TEST_S3_BUCKET}/{TEST_S3_PATH}/', f'{LOCAL_DATA_DIR}/')


def localisation_inference(model_col, params, training_name):
    real_path = f'{LOCAL_DATA_DIR}/labels/'
    pred_path = f'{LOCAL_DATA_DIR}/model_outputs/'
    test_data_df = pd.read_csv(f'test_set_v1.csv')  # Read test data csv
    os.makedirs(pred_path, exist_ok=True)

    print('Fetching model output from lambda.')
    get_model_output_multiprocessing(model_col)  # Get test set predictions for current model.
    print('Completed model output from lambda.')

    print('Starting getting model score.')
    thresh_iou = [0.5]
    model_result = get_score(test_data_df, real_path, pred_path, thresh_iou)  # Get score for current model's prediction
    mail_body = get_mail_body(model_result)  # Check if current model score is better than best model score
    print('Completed getting model score.')

    print('Starting bucket analysis.')
    bucket_result, bucket_output = get_bucket_analysis(test_data_df, real_path, pred_path, thresh_iou)  # Get bucket analysis for current model
    print('Completed bucket analysis.')

    print('Starting sending email.')
    final_output = str(params) + '<br>' + mail_body + '<br>' + bucket_output  # Combine all result to be sent on mail
    call_email_lambda(final_output, training_name)  # Send all the results by mail
    print('Completed sending email.')

    return None
