import json
import pandas as pd
from global_variables import LOCAL_DATA_DIR


def intersection_over_union_2D(gt_box, pred_box):
    """
        Function to calculate the IOU score for 2D-boxes.
        gt_box: Ground truth bounding box. [left, top, right, bottom]
        pred_box: Predicted bounding box. [left, top, right, bottom]
    """
    inter_box_left_top = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_right_bottom = [min(gt_box[2], pred_box[2]), min(gt_box[3], pred_box[3])]

    inter_box_w = inter_box_right_bottom[0] - inter_box_left_top[0]
    inter_box_h = inter_box_right_bottom[1] - inter_box_left_top[1]

    intersection = inter_box_w * inter_box_h
    union = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) + (pred_box[2] - pred_box[0]) * (
            pred_box[3] - pred_box[1]) - intersection

    iou = intersection / union
    if iou<0:
        return 0

    return iou


def intersection_over_union_1D(real_line, pred_line):
    """
        Function to calculate the IOU score for 1D-lines.
        real_line: Ground truth lines. [x1, x2]
        pred_line: Predicted lines. [x1, x2]
    """
    a1 = real_line[0]
    a2 = real_line[1]
    b1 = pred_line[0]
    b2 = pred_line[1]

    # In case line 2 is completely within line 1
    if a1 < b1 < a2 and a1 < b2 < a2:
        return (b2 - b1) / (a2 - a1)

    if a1 > b1:
        return intersection_over_union_1D(pred_line, real_line)

    # In case of overlap
    if a2 >= b1:
        return (a2 - b1) / (b2 - a1)
    else:  # In case of non-overlap
        return 0


def intersection_over_union(real, pred, dimension):
    """
        Function to call 1D-IOU or 2D-IOU as required.
    """
    if dimension == '1D':
        return intersection_over_union_1D(real, pred)
    elif dimension == '2D':
        return intersection_over_union_2D(real, pred)


def match_prediction(real_col_lines, pred_line, real_col_lines_match, thresh_iou, D, return_line=False):
    """
        Function to match the Predicted Line with all the ground truth lines.
        real_col_lines: List of ground truth lines. [[x1, x2], [x1, x2].......]
        pred_line: The current prediction line to match with the list of real_col_lines. [x1, x2]
        real_col_lines_match: List of Bool. It denotes if a particular ground truth line has been matched or not. [Bool, Bool, .......]
        thresh_iou: FLOAT. The minimum IOU score which the predicted line must have with the ground truth line to be considered as a prediction.
        D: "1D/2D". It denoted which type of IOU we want to use.
    """
    max_score = 0
    idx = -1

    for i, real_line in enumerate(real_col_lines):
        if real_col_lines_match[i] == 1:
            continue
        score = intersection_over_union(real_line, pred_line, D)
        if score >= max_score:
            max_score = score
            idx = i

    if max_score < thresh_iou:
        if return_line:
            return False, None
        else:
            return False
    else:
        real_col_lines_match[idx] = 1
        if return_line:
            return True, idx
        else:
            return True


def precision_recall(TP, FP, FN):
    """
        Function to calculate the precision and recall from TP, FP and FN.
        TP: Number of True Positives.
        FP: Number of False Positives.
        FN: Number of False Negative.
    """
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def read_ocr(filename, page_no, real_path):
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    real_table_coord = list(map(float, real_json['table_coordinates'].split(',')))
    l, t, r, b = real_table_coord

    ocr_path = f'{LOCAL_DATA_DIR}/ocr/{filename}.parquet'
    df_ocr = pd.read_parquet(ocr_path)
    df_ocr = df_ocr[(df_ocr['page'] == page_no) & (df_ocr['minx']>=l) & (df_ocr['maxx']<=r) & (df_ocr['miny']>=t) & (df_ocr['maxy']<=b)]
    df_ocr.loc[:, 'midx'] = (df_ocr['minx'] + df_ocr['maxx']) / 2
    df_ocr.loc[:, 'midy'] = (df_ocr['miny'] + df_ocr['maxy']) / 2
    return df_ocr


def check_in_between(df_ocr, intersection, is_column):
    p1, p2 = intersection
    if is_column:
        words_between = df_ocr[(df_ocr['midx'] >= p1) & (df_ocr['midx'] <= p2)]
    else:
        words_between = df_ocr[(df_ocr['midy'] >= p1) & (df_ocr['midy'] <= p2)]
    if words_between.empty:
        return False
    else:
        return True


def check_char(real_line, pred_line, df_ocr, is_column):

    intersection_1 = [min(real_line[0], pred_line[0]), max(real_line[0], pred_line[0])]
    intersection_2 = [min(real_line[1], pred_line[1]), max(real_line[1], pred_line[1])]

    tot = check_in_between(df_ocr, intersection_1, is_column) + check_in_between(df_ocr, intersection_2, is_column)
    if tot > 0:
        return False
    return True


def confusion_matrix(real, pred, thresh_iou, do_char_check, df_ocr, is_column):
    """
        Function which gets the list of ground truth and list of predicted lines as input and returns the number of TP, FP and FN.
        real: List of ground truth lines.
        pred: List of predicted lines.
        thresh_iou: FLOAT. The minimum IOU score which the predicted line must have with the ground truth line to be considered as a prediction.
    """
    real_col_lines = [[real[i], real[i + 1]] for i in range(len(real) - 1)]
    pred_col_lines = [[pred[i], pred[i + 1]] for i in range(len(pred) - 1)]

    real_col_lines_match = [0 for _ in range(len(real_col_lines))]
    TP = 0
    FP = 0
    for pred_line in pred_col_lines:
        match, idx = match_prediction(real_col_lines, pred_line, real_col_lines_match, thresh_iou, "1D", return_line=True)
        if match and do_char_check:
            val = check_char(real_col_lines[idx], pred_line, df_ocr, is_column)
            if val:
                TP += 1
            else:
                FP += 1
                real_col_lines_match[idx] = 0
        elif match:
            TP += 1
        else:
            FP += 1
    FN = len(real_col_lines) - sum(real_col_lines_match)

    return TP, FP, FN


def table_score(real_table_coord, pred_table_coord):
    """
        Function to calculate the IOU score for the predicted table.
        real_table_coord: Ground truth table bounding box. [left, top, right, bottom]
        pred_table_coord: Predicted table bounding box. [left, top, right, bottom]
    """
    return intersection_over_union(real_table_coord, pred_table_coord, "2D")


def get_real_json(prefix, name):
    """
        Function to read the json files containing the predictions.
        prefix: Path where all the predictions (json files) are stored.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    f = open(prefix + name)
    data = json.load(f)
    f.close()

    return data


def metrics_table(real_path, pred_path, filename):
    """
        Function to read the ground truth and prediction json files and return the table score.
        real_path: Path to the ground truth json files.
        pred_path: Path to the prediction json files.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    pred_json = get_real_json(pred_path, name)

    real_table_coord = list(map(float, real_json['table_coordinates'].split(',')))
    if not pred_json['table_coordinates']:
        return 0
    pred_table_coord = list(map(float, pred_json['table_coordinates'].split(',')))

    return table_score(real_table_coord, pred_table_coord)


def metrics_col(real_path, pred_path, filename, thresh_iou, df_ocr):
    """
        Function to read the ground truth and prediction json files and return the confusion matrix (TP, FP, FN, precision, recall) for the column seprators.
        real_path: Path to the ground truth json files.
        pred_path: Path to the prediction json files.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    pred_json = get_real_json(pred_path, name)

    real_table_coord = list(map(float, real_json['table_coordinates'].split(',')))
    if pred_json['table_coordinates']:
        pred_table_coord = list(map(float, pred_json['table_coordinates'].split(',')))

    real_column_sep = list(map(float, real_json['column_separators'].split(',')))

    if not pred_json['column_separators']:  # no col seps detected
        TP, FP, FN = 0, 0, len(real_column_sep)
        return TP, FP, FN
    pred_column_sep = list(map(float, pred_json['column_separators'].split(',')))

    real_column_sep = [real_table_coord[0]] + real_column_sep + [real_table_coord[2]]
    pred_column_sep = [pred_table_coord[0]] + pred_column_sep + [pred_table_coord[2]]

    return confusion_matrix(real_column_sep, pred_column_sep, thresh_iou, True, df_ocr, is_column=True)


def metrics_row(real_path, pred_path, filename, thresh_iou, df_ocr):
    """
        Function to read the ground truth and prediction json files and return the confusion matrix (TP, FP, FN, precision, recall) for the row seprators.
        real_path: Path to the ground truth json files.
        pred_path: Path to the prediction json files.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    pred_json = get_real_json(pred_path, name)

    real_table_coord = list(map(float, real_json['table_coordinates'].split(',')))
    pred_table_coord = list(map(float, real_json['table_coordinates'].split(',')))

    real_row_sep = list(map(float, real_json['row_separators'].split(',')))

    if not pred_json['row_separators']:
        TP, FP, FN = 0, 0, len(real_row_sep)
        return TP, FP, FN
    pred_row_sep = list(map(float, pred_json['row_separators'].split(',')))

    real_row_sep = [real_table_coord[1]] + real_row_sep + [real_table_coord[3]]
    pred_row_sep = [pred_table_coord[1]] + pred_row_sep + [pred_table_coord[3]]

    return confusion_matrix(real_row_sep, pred_row_sep, thresh_iou, True, df_ocr, is_column=False)


def check_table(real_path, pred_path, filename):
    """
        Function to check if the table exits in the ground truth and the predictions.
        real_path: Path to the ground truth json files.
        pred_path: Path to the prediction json files.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    pred_json = get_real_json(pred_path, name)
    return 'table_coordinates' in real_json.keys() and 'table_coordinates' in pred_json.keys()


def check_column(real_path, pred_path, filename):
    """
        Function to check if the column seprators exits in the ground truth and the predictions.
        real_path: Path to the ground truth json files.
        pred_path: Path to the prediction json files.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    pred_json = get_real_json(pred_path, name)
    return 'column_separators' in real_json.keys() and 'column_separators' in pred_json.keys()


def check_row(real_path, pred_path, filename):
    """
        Function to check if the row seprators exits in the ground truth and the predictions.
        real_path: Path to the ground truth json files.
        pred_path: Path to the prediction json files.
        doc_id: The Doc_ID of the image for which we want the prediction for.
        page_no: The Page number of the image in the pdf for which we want the predictions for.
    """
    name = filename + '.json'
    real_json = get_real_json(real_path, name)
    pred_json = get_real_json(pred_path, name)
    return 'row_separators' in real_json.keys() and 'row_separators' in pred_json.keys()
