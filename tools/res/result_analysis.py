import cPickle
import numpy as np
from pycocotools import mask as maskUtils
from operator import itemgetter, attrgetter

def summarize_mAP(iouThr=None, areaRng='all', maxDets=100 ):
    '''
    Compute mAP under different scale
    '''
    print('''
    Compute mAP under different scale
    ''')
    print("{:16}\t{:16}\t{}".format("Average  Precision", "Average Recall", "Condition"))
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    # precision
    # dimension of precision: [TxRxKxAxM]
    t_p = eval_result.eval['precision']
    # IoU
    if iouThr is not None:
        t = np.where(iouThr == p.iouThrs)[0]
        t_p = t_p[t]
    t_p = t_p[:,:,:,aind,mind]
    if len(t_p[t_p>-1])==0:
        ap = -1
    else:
        ap = np.mean(t_p[t_p>-1])
    # recall
    # dimension of recall: [TxKxAxM]
    t_r = eval_result.eval['recall']
    if iouThr is not None:
        t = np.where(iouThr == p.iouThrs)[0]
        t_r = t_r[t]
    t_r = t_r[:,:,aind,mind]
    if len(t_r[t_r>-1])==0:
        ar = -1
    else:
        ar = np.mean(t_r[t_r>-1])
    condition_str = "@[areaRng={} | maxDets={}]".format(areaRng, maxDets)
    print("{:16}\t{:16}\t{}".format(ap, ar, condition_str))

def summarize_areaRng_scoresThrs(scores_thrs):
    '''
    Compute different areaRng (small, medium, large) objects' precision & recall 
    under 0.7 scores
    '''
    print('''
    Compute different areaRng (small, medium, large) objects' precision & recall 
    under 0.7 scores
    ''')
    print("{:16}\t{:16}\t{}".format("Precision", "Recall", "Condition"))
    scores = eval_result.eval['scores']
    scores_mask = scores >= scores_thrs # 0.7 score threshold
    scores_mask = scores_mask.astype(int)
    maxDets_ind = 2 # 100 maxDets
    for areaRng_ind in range(1,len(p.areaRngLbl)):
        # precision
        # dimension of precision: [TxRxKxAxM]
        t_precision = eval_result.eval['precision']
        t_precision = np.multiply(t_precision, scores_mask)
        t_precision = t_precision[:,:,:,areaRng_ind,maxDets_ind]
        t_precision = t_precision[t_precision>0]
        if len(t_precision[t_precision>-1])==0:
            ap = -1
        else:
            ap = np.mean(t_precision[t_precision>-1])
        # recall
        # dimension of recall: [TxKxAxM]
        # t_recall = np.multiply(recall,scores_mask[:,1,:,:,:]) #cocoeval.py line 411
        t_recall = eval_result.eval['recall']
        t_recall = np.multiply(t_recall, np.mean(scores_mask, axis=1)) #cocoeval.py line 411
        t_recall = t_recall[:,:,areaRng_ind,maxDets_ind]
        t_recall = t_recall[t_recall>0]
        if len(t_recall[t_recall>-1])==0:
            ar = -1
        else:
            ar = np.mean(t_recall[t_recall>-1])
        condition_str = "@ [areaRng = {} | scores >= {} | maxDets = 100]". \
            format(p.areaRngLbl[areaRng_ind], scores_thrs)
        print("{:16}\t{:16}\t{}".
            format(ap, ar, condition_str))


def summarize_cat_iou():
    '''
    Compute different category objects' precision & recall 
    under different IoU
    '''
    print('''
    Compute different category objects' precision & recall 
    under different IoU
    ''')
    print("{:8}\t{:16}\t{:16}\t{}".format("ClassName", "Precision", "Recall", "Condition"))
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    for ind_lo in range(0,len(p.iouThrs)):
        for cat_ind in range(0,len(p.catIds)):
            # precision
            t_precision = eval_result.eval['precision']
            t_precision = t_precision[ind_lo:len(p.iouThrs), :, cat_ind - 1, 0, 2]
            ap = np.mean(t_precision[t_precision > -1])
            # recall
            t_recall = eval_result.eval['recall']
            t_recall = t_recall[ind_lo:len(p.iouThrs), cat_ind - 1, 0, 2]
            ar = np.mean(t_recall[t_recall > -1])
            
            iou_str = "{}:{}".format(p.iouThrs[ind_lo], p.iouThrs[len(p.iouThrs)-1])
            condition_str = "@ [catIds = {} | IoU = {}]".format(p.catIds[cat_ind], iou_str)
            print("{:8}\t{:16}\t{:16}\t{}".
                format(cat_name[cat_ind]['name'], ap, ar, condition_str))

def summarize_errorType(ious_thrs):
    '''
    Compute the ratio of prediect correct V.S. catgeory error V.S. localization error
    for different category objects' 
    '''
    print('''
    Compute the ratio of prediect correct V.S. catgeory error V.S. localization error
    for different category objects' 
    ''')
    print("IoU threshold is {}".format(ious_thrs))
    print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
        format("ClassName", "Detection Number", "Correct", "Label miss", "Category error", "Localization error"))
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    def is_intersec_other(all_ious, dt_ind, all_gt, catId):
        for gt_ind in range(0, all_ious.shape[1]):
            if (all_ious[dt_ind, gt_ind] > ious_thrs) and \
                (catId != all_gt[gt_ind]['category_id']):
                return True
        return False
    
    cat_ind = 0
    for catId in p.catIds:
        dt_count = 0
        correct = 0
        label_miss = 0
        loc_error = 0
        cat_error = 0
        #print("run in catId {}".format(catId))
        for imgId in p.imgIds:
            # (per cat & per img)'s gt & dt
            dt = eval_result._dts[imgId, catId]
            gt = eval_result._gts[imgId, catId]
            # all groudtruth in this image
            all_gt = [temp_gt_per_cat for temp_catId in p.catIds for temp_gt_per_cat in eval_result._gts[imgId, temp_catId]]
            # all category appeared in this image's gt
            all_cat = [temp_gt["category_id"] for temp_gt in all_gt]
            # all iou cross cat per dt            
            g = [temp_gt['bbox'] for temp_gt in all_gt]
            d = [temp_dt['bbox'] for temp_dt in dt]
            iscrowd = [int(o['iscrowd']) for o in all_gt]
            dtgt_ious = eval_result.ious[imgId, catId] # shape = ndarray(dt_num, gt_num)
            all_ious = maskUtils.iou(d,g,iscrowd) # shape = ndarray(d_num, g_num)
            dt_count = dt_count + len(dt)
            # label miss
            if len(dt) > 0 and not (catId in all_cat):
                label_miss = label_miss + len(dt)
                continue
            if len(dtgt_ious) <= 0:
                loc_error = loc_error + len(dt)
            else:
                for dt_ind in range(0, dtgt_ious.shape[0]):
                    has_correct = False
                    has_cat_error = False
                    has_loc_error = False
                    for gt_ind in range(0, dtgt_ious.shape[1]):
                        if (not has_correct):
                            if (dtgt_ious[dt_ind, gt_ind] > ious_thrs) and \
                                (dt[dt_ind]['category_id'] == gt[gt_ind]['category_id']):
                                has_correct = True
                                break
                            # localization error
                            if (dtgt_ious[dt_ind, gt_ind] < ious_thrs) and \
                                (dt[dt_ind]['category_id'] == gt[gt_ind]['category_id']):
                                # category error
                                if is_intersec_other(all_ious, dt_ind, all_gt, catId):
                                    has_cat_error = True
                                else:
                                    has_loc_error = True
                    if has_correct:
                        correct = correct + 1
                    elif has_cat_error:
                        cat_error = cat_error + 1
                    elif has_loc_error:
                        loc_error = loc_error + 1
        dt_count = float(dt_count)
        r_correct = correct / dt_count
        r_cat_error = cat_error / dt_count
        r_loc_error = loc_error / dt_count
        r_label_miss = label_miss / dt_count
        print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
            format(cat_name[cat_ind]['name'], dt_count, r_correct, r_label_miss, r_cat_error, r_loc_error))
        cat_ind = cat_ind + 1

def summarize_errorType_scale(ious_thrs):
    '''
    Compute the ratio of prediect correct V.S. catgeory error V.S. localization error
    for different category objects' in multi scale
    '''
    print('''
    Compute the ratio of prediect correct V.S. catgeory error V.S. localization error
    for different category objects' in multi scale
    ''')
    print("IoU threshold is {}".format(ious_thrs))
    print("{:16}\t{:16}\t{:48}\t{:48}\t{:48}\t{:48}".
        format("ClassName", "Detection Number", "Correct", "Label miss", "Category error", "Localization error"))
    print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
        format("","","small","medium","large","small","medium","large","small","medium","large","small","medium","large"))
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    def is_intersec_other(all_ious, dt_ind, all_gt, catId):
        for gt_ind in range(0, all_ious.shape[1]):
            if (all_ious[dt_ind, gt_ind] > ious_thrs) and \
                (catId != all_gt[gt_ind]['category_id']):
                return True
        return False
    
    cat_ind = 0
    scale = p.areaRng
    def get_scale_ind(area):
        for i in range(1,len(scale)):
            if area > scale[i][0] and area < scale[i][1]:
                return i
        return -1 

    total = {"dt_count": 0.0, "correct": np.zeros(3), "cat_error": np.zeros(3), "loc_error": np.zeros(3), "label_miss": np.zeros(3)}
    for catId in p.catIds:
        dt_count = 0
        correct = np.zeros(3) # small medium large
        loc_error = np.zeros(3)
        label_miss = np.zeros(3)
        cat_error = np.zeros(3)
        #print("run in catId {}".format(catId))
        for imgId in p.imgIds:
            # (per cat & per img)'s gt & dt
            dt = eval_result._dts[imgId, catId]
            gt = eval_result._gts[imgId, catId]
            # all groudtruth in this image
            all_gt = [temp_gt_per_cat for temp_catId in p.catIds for temp_gt_per_cat in eval_result._gts[imgId, temp_catId]]
            # all category appeared in this image's gt
            all_cat = [temp_gt["category_id"] for temp_gt in all_gt]
            # all iou cross cat per dt            
            g = [temp_gt['bbox'] for temp_gt in all_gt]
            d = [temp_dt['bbox'] for temp_dt in dt]
            iscrowd = [int(o['iscrowd']) for o in all_gt]
            dtgt_ious = eval_result.ious[imgId, catId] # shape = ndarray(dt_num, gt_num)
            all_ious = maskUtils.iou(d,g,iscrowd) # shape = ndarray(d_num, g_num)
            dt_count = dt_count + len(dt)
            # label miss
            if len(dt) > 0 and not (catId in all_cat):
                #label_miss = label_miss + len(dt)
                for temp_dt in dt:
                    scale_ind = get_scale_ind(temp_dt['area']) - 1
                    label_miss[scale_ind] = label_miss[scale_ind] + 1
                continue
            if len(dtgt_ious) <= 0:
                loc_error = loc_error + len(dt)
            else:
                for dt_ind in range(0, dtgt_ious.shape[0]):
                    has_correct = False
                    has_cat_error = False
                    has_loc_error = False
                    for gt_ind in range(0, dtgt_ious.shape[1]):
                        if (not has_correct):
                            if (dtgt_ious[dt_ind, gt_ind] > ious_thrs) and \
                                (dt[dt_ind]['category_id'] == gt[gt_ind]['category_id']):
                                has_correct = True
                                break
                            # localization error
                            if (dtgt_ious[dt_ind, gt_ind] < ious_thrs) and \
                                (dt[dt_ind]['category_id'] == gt[gt_ind]['category_id']):
                                # category error
                                if is_intersec_other(all_ious, dt_ind, all_gt, catId):
                                    has_cat_error = True
                                else:
                                    has_loc_error = True
                    scale_ind = get_scale_ind(dt[dt_ind]['area']) - 1
                    if has_correct:
                        correct[scale_ind] = correct[scale_ind] + 1
                    elif has_cat_error:
                        cat_error[scale_ind]= cat_error[scale_ind] + 1
                    elif has_loc_error:
                        loc_error[scale_ind] = loc_error[scale_ind] + 1
        total["dt_count"] = total["dt_count"] + dt_count
        total["correct"] = total["correct"] + correct
        total["cat_error"] = total["cat_error"] + cat_error
        total["loc_error"] = total["loc_error"] + loc_error
        total["label_miss"] = total["label_miss"] + label_miss
        dt_count = float(dt_count)
        r_correct = correct / dt_count
        r_cat_error = cat_error / dt_count
        r_loc_error = loc_error / dt_count
        r_label_miss = label_miss / dt_count
        print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
            format(cat_name[cat_ind]['name'], dt_count, r_correct[0], r_correct[1], r_correct[2], r_label_miss[0], r_label_miss[1], r_label_miss[2], r_cat_error[0], r_cat_error[1], r_cat_error[2], r_loc_error[0], r_loc_error[1], r_loc_error[2]))
        cat_ind = cat_ind + 1

    print("===========================================")
    total["dt_count"] = float(total["dt_count"])
    total["correct"] = total["correct"] / total["dt_count"]
    total["cat_error"] = total["cat_error"] / total["dt_count"]
    total["loc_error"] = total["loc_error"] / total["dt_count"]
    total["label_miss"] = total["label_miss"] / total["dt_count"]
    print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
        format("All Class", total["dt_count"], total["correct"][0], total["correct"][1], total["correct"][2], total["label_miss"][0], total["label_miss"][1], total["label_miss"][2], total["cat_error"][0], total["cat_error"][1], total["cat_error"][2], total["loc_error"][0], total["loc_error"][1], total["loc_error"][2]))

def summarize_finerErrorType_scale(score_thrs, iou_rng):
    '''
    Compute the ratio of prediect correct V.S. class error V.S. localization error V.S. BG
    for different category objects' in multi scale
    '''
    print('''
    Compute the ratio of prediect correct V.S. class error V.S. localization error V.S. BG
    for different category objects' in multi scale
    ''')
    ious_lo = iou_rng[0]
    ious_hi = iou_rng[1]
    print("IoU threshold is between {} and {}".format(ious_lo, ious_hi))
    print("Score threshold is {}".format(score_thrs))
    print("{:16}\t{:16}\t{:48}\t{:48}\t{:48}\t{:48}".
        format("ClassName", "Detection Number", "Correct", "Class error", "Localization error", "BG trap"))
    print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
        format("","","small","medium","large","small","medium","large","small","medium","large","small","medium","large"))
    
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    cat_ind = 0
    def is_intersec_other(all_ious, dt_ind, all_gt, catId):
        for gt_ind in range(0, all_ious.shape[1]):
            if (all_ious[dt_ind, gt_ind] > ious_hi) and \
                (catId != all_gt[gt_ind]['category_id']):
                return True
        return False
    
    scale = p.areaRng
    def get_scale_ind(area):
        for i in range(1,len(scale)):
            if area > scale[i][0] and area < scale[i][1]:
                return i
        return -1 

    total = {"dt_count": 0.0, "correct": np.zeros(3), "cat_error": np.zeros(3), "loc_error": np.zeros(3), "bg_trap": np.zeros(3)}
    all_cat_total = []
    all_cat_total_wto_scale = []
    for catId in p.catIds:
        dt_count = 0
        correct = np.zeros(3) # small medium large
        cat_error = np.zeros(3)
        loc_error = np.zeros(3)
        bg_trap = np.zeros(3)
        #print("run in catId {}".format(catId))
        for imgId in p.imgIds:
            # (per cat & per img)'s gt & dt
            dt = eval_result._dts[imgId, catId]
            gt = eval_result._gts[imgId, catId]
            # all groudtruth in this image
            all_gt = [temp_gt_per_cat for temp_catId in p.catIds for temp_gt_per_cat in eval_result._gts[imgId, temp_catId]]
            # all category appeared in this image's gt
            all_cat = [temp_gt["category_id"] for temp_gt in all_gt]
            # all iou cross cat per dt            
            g = [temp_gt['bbox'] for temp_gt in all_gt]
            d = [temp_dt['bbox'] for temp_dt in dt]
            iscrowd = [int(o['iscrowd']) for o in all_gt]
            dtgt_ious = eval_result.ious[imgId, catId] # shape = ndarray(dt_num, gt_num)
            all_ious = maskUtils.iou(d,g,iscrowd) # shape = ndarray(d_num, g_num)
            for dt_ind in range(0, len(dtgt_ious)):
                if dt[dt_ind]['score'] < score_thrs:
                    continue
                has_correct = False
                has_cat_error = False
                has_loc_error = False
                has_bg_trapped = False
                for gt_ind in range(0, dtgt_ious.shape[1]):
                    if (not has_correct):
                        if (dtgt_ious[dt_ind, gt_ind] >= ious_hi):
                            if (dt[dt_ind]['category_id'] == gt[gt_ind]['category_id']):
                                # correct
                                has_correct = True
                                break
                            else:
                                # class error
                                has_cat_error = True
                        if (dtgt_ious[dt_ind, gt_ind] < ious_hi) and (dtgt_ious[dt_ind, gt_ind] >= ious_lo):
                            if (dt[dt_ind]['category_id'] == gt[gt_ind]['category_id']):
                                # localization error
                                has_loc_error = True
                            elif not is_intersec_other(all_ious, dt_ind, all_gt, catId):
                                # bg trap
                                has_bg_trapped = True
                            else:
                                # class error
                                has_cat_error = True
                        if (dtgt_ious[dt_ind, gt_ind] < ious_lo):
                            if not is_intersec_other(all_ious, dt_ind, all_gt, catId):
                                # bg trap
                                has_bg_trapped = True
                            else:
                                # class error
                                has_cat_error = True
                dt_count = dt_count + 1
                scale_ind = get_scale_ind(dt[dt_ind]['area']) - 1
                if has_correct:
                    correct[scale_ind] = correct[scale_ind] + 1
                elif has_cat_error:
                    cat_error[scale_ind]= cat_error[scale_ind] + 1
                elif has_loc_error:
                    loc_error[scale_ind] = loc_error[scale_ind] + 1
                elif has_bg_trapped:
                    bg_trap[scale_ind] = bg_trap[scale_ind] + 1
        total["dt_count"] = total["dt_count"] + dt_count
        total["correct"] = total["correct"] + correct
        total["cat_error"] = total["cat_error"] + cat_error
        total["loc_error"] = total["loc_error"] + loc_error
        total["bg_trap"] = total["bg_trap"] + bg_trap
        dt_count = float(dt_count)
        r_correct = correct / dt_count
        r_cat_error = cat_error / dt_count
        r_loc_error = loc_error / dt_count
        r_bg_trap = bg_trap / dt_count
        all_cat_total.append((cat_name[cat_ind]['name'], np.sum(correct)/dt_count, dt_count, r_correct, r_cat_error, r_loc_error, r_bg_trap))
        all_cat_total_wto_scale.append((cat_name[cat_ind]['name'], dt_count, np.sum(correct)/dt_count, np.sum(cat_error)/dt_count, np.sum(loc_error)/dt_count, np.sum(bg_trap)/dt_count))
        print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
            format(cat_name[cat_ind]['name'], dt_count, r_correct[0], r_correct[1], r_correct[2], r_cat_error[0], r_cat_error[1], r_cat_error[2], r_loc_error[0], r_loc_error[1], r_loc_error[2], r_bg_trap[0], r_bg_trap[1], r_bg_trap[2]))
        cat_ind = cat_ind + 1

    print("===========================================")
    total["dt_count"] = float(total["dt_count"])
    total["correct"] = total["correct"] / total["dt_count"]
    total["cat_error"] = total["cat_error"] / total["dt_count"]
    total["loc_error"] = total["loc_error"] / total["dt_count"]
    total["bg_trap"] = total["bg_trap"] / total["dt_count"]
    print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
        format("All Class", total["dt_count"], total["correct"][0], total["correct"][1], total["correct"][2], total["cat_error"][0], total["cat_error"][1], total["cat_error"][2], total["loc_error"][0], total["loc_error"][1], total["loc_error"][2], total["bg_trap"][0], total["bg_trap"][1], total["bg_trap"][2]))
    print("=============== sorted  ==================")
    all_cat_total = sorted(all_cat_total, key=itemgetter(1))
    for item in all_cat_total:
        print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
            format(item[0], item[2], item[3][0], item[3][1], item[3][2], item[4][0], item[4][1], item[4][2], item[5][0], item[5][1], item[5][2], item[6][0], item[6][1], item[6][2]))
    print("=============== sorted wt.o scale ==================")
    all_cat_total_wto_scale = sorted(all_cat_total_wto_scale, key=itemgetter(2))
    for item in all_cat_total_wto_scale:
        print("{:16}\t{:16}\t{:16}\t{:16}\t{:16}\t{:16}".
            format(item[0], item[1], item[2], item[3], item[4], item[5]))

def count_cat_gtbbox():
    '''
    Count the total gt number of each category bbox
    '''
    print('''
    Count the total gt number of each category bbox 
    ''')
    print("{:16}\t{:16}".format("ClassName", "#Gt bbox"))
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    cat_ind = 0
    for catId in p.catIds:
        gt_count = 0
        # all groudtruth in this category
        all_gt = [temp_gt_per_cat for temp_imgId in p.imgIds for temp_gt_per_cat in eval_result._gts[temp_imgId, catId]]
        # all category appeared in this image's gt
        gt_count = len(all_gt)
        print("{:16}\t{:16}".format(cat_name[cat_ind]['name'], gt_count))
        cat_ind = cat_ind + 1


# def summarize_proposal_recall():
#     '''
#     Compute recall of iou
#     '''
#     #(365387, 70), (527528, 51), (370270, 42), (99053, 51), (217425, 80), (396338, 11), (319369, 58), (503755, 74), (137246, 82), (170099, 28), (414133, 21), (84664, 39), (486040, 80), (177015, 72), (47828, 20), (463690, 20), (475365, 54), (239347, 33), (243989, 5), (345261, 87), (196759, 49), (100510, 20), (25228, 7), (547854, 24), (211120, 65), (368961, 20), (106912, 72), (142585, 49), (266768, 41), (58029, 90), (138550, 64), (155571, 79), (14038, 82), (52413, 2), (456394, 73), (234779, 47), (110784, 73), (230166, 74), (26941, 58), (323263, 82), (210030, 60), (246454, 15), (121417, 3), (122745, 77), (81766, 25), (191672, 56), (86755, 39), (525083, 59), (481480, 78), (84492, 79), (262048, 7), (52891, 25), (574425, 6), (231879, 39), (427034, 8), (552371, 22), (356498, 53), (532058, 38), (306733, 21), (208363, 59), (81594, 56), (320664, 19), (172547, 89), (374551, 86), (120584, 64), (344909, 76), (334719, 87), (474293, 31), (118594, 1), (269316, 53), (389684, 18), (143998, 57)
# summarize_iou_recall()

def count_cat_gtdtImg():
    '''
    Count the total image number where a category gt | dt appears
    '''
    print('''
    Count the total image number where a category gt | dt appears
    ''')
    print("======= gt ========")
    print("{:16}\t{:16}".format("ClassName", "#image"))
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    cat_ind = 0
    for catId in p.catIds:
        img_count = 0
        # all images where a category's gt appear
        all_img_id = [temp_gt['image_id'] for temp_imgId in p.imgIds for temp_gt in eval_result._gts[temp_imgId, catId]]
        distinct_img_id = list(set(all_img_id))
        img_count = len(distinct_img_id)
        print("{:16}\t{:16}".format(cat_name[cat_ind]['name'], img_count))
        cat_ind = cat_ind + 1

    print("======= dt ========")
    print("{:16}\t{:16}".format("ClassName", "#image"))
    cat_name = eval_result.cocoGt.loadCats(eval_result.cocoGt.getCatIds())
    cat_ind = 0
    for catId in p.catIds:
        img_count = 0
        # all images where a category's gt appear
        all_img_id = [temp_gt['image_id'] for temp_imgId in p.imgIds for temp_gt in eval_result._gts[temp_imgId, catId]]
        distinct_img_id = list(set(all_img_id))
        img_count = len(distinct_img_id)
        print("{:16}\t{:16}".format(cat_name[cat_ind]['name'], img_count))
        cat_ind = cat_ind + 1

if __name__ == "__main__":
    result_file = "detection_results.pkl"
    print("start result analysis from " + result_file)
    eval_result = cPickle.load(open(result_file,"rb"))
    #print(eval_result)
    p = eval_result.params
    # print(p.iouThrs)
    # print("T=10 IoU thresholds: {}".format(len(p.iouThrs)))
    # print(p.recThrs)
    # print(len(p.recThrs))
    # print("R=101 recall thresholds: {}".format(len(p.recThrs)))
    # print(p.catIds)
    # print("K cat ids: {}".format(len(p.catIds)))
    # print(p.areaRng)
    # print("A=4 area ranges: {}".format(len(p.areaRng)))
    # print(p.maxDets)
    # print("M=3 thresholds on max detections: {}".format(len(p.maxDets)))
    # 
    # precision = eval_result.eval['precision']
    # recall = eval_result.eval['recall']
    # scores = eval_result.eval['scores']
    # print("precision's shape is {}".format(precision.shape))
    # print("recall's shape is {}".format(recall.shape))
    # print("scores's shape is {}".format(scores.shape))
    
    #summarize_mAP(areaRng='small')
    #summarize_mAP(areaRng='medium')
    #summarize_mAP(areaRng='large')
    
    #summarize_areaRng_scoresThrs(0.5)
    #summarize_areaRng_scoresThrs(0.7)
    
    #summarize_cat_iou()

    #summarize_errorType(0.5)
    #summarize_errorType_scale(0.5)
    for score_thrs in [0.5, 0.7]:
        summarize_finerErrorType_scale(score_thrs, [0.1, 0.5])

    #count_cat_gtbbox()

    #count_cat_gtdtImg()
