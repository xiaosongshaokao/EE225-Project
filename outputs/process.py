import cv2 as cv
import numpy as np

def DealWiFi(count, j):
    img_path = '../recombination/' + str(j) + '.jpg'
    img = cv.imread(img_path)
    img1 = img.copy()
    img2 = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # gray = cv.morphologyEx(gray)
    #平滑滤波
    # gray = cv.blur(gray, (5, 5))#均值滤波
    # gray = cv.fastNlMeansDenoising(gray,None, 10, 7, 21)
    # gray = cv.GaussianBlur(gray, (5, 5), 0)#高斯滤波
    boxes = []
    for threshold in range(175, 135, -1):
        #二值化
        ret, mat = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
        # kernel = np.ones((3, 3), np.uint8)
        # mat = cv.morphologyEx(mat, cv.MORPH_OPEN, kernel, iterations=2)
        mat = cv.medianBlur(mat, 5)
        cv.imshow('mat', mat)
        cv.waitKey()
        #找出二值图中的外边界
        contours, hierarchy = cv.findContours(mat, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = box.astype(int)
            if abs((box[1][0] - box[0][0]) - (box[3][0] - box[2][0])) > 20:
                continue
            Xs = [k[0] for k in box]
            Ys = [k[1] for k in box]
            x1 = abs(min(Xs))
            x2 = abs(max(Xs))
            y1 = abs(min(Ys))
            y2 = abs(max(Ys))
            hight = y2 - y1
            width = x2 - x1
            if width == 0 or hight == 0:
                continue
            #对外边框进行筛选，以得到可能是WiFi的边框
            if (hight >= 130 and hight <= 300) or (150 <= width and width <= 250):
                tmp = img.copy()
                cv.drawContours(img, [box.astype(int)], 0, (255, 255, 255), 1)
                cv.imshow('1', tmp)

                boxes.append([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            # else:
            #     continue
    cv.imshow('original', img)
    cv.waitKey()
    #外边框排序
    boxes.sort(key = lambda x:x[0][0], reverse=False)
    # print(boxes)
    index = 0
    single = []
    flag = 0
    multi = []
    #分离多次出现和单独出现的边框
    while index < len(boxes) - 1:
        if index == len(boxes) - 2:
            left = boxes[index]
            right = boxes[index + 1]
            if right[0][0] - left[0][0] < 20:
                if abs(left[1][0] - right[1][0]) < 20:
                    multi.append(left)
                    multi.append(right)
                else:
                    single.append(left)
            else:
                single.append(left)
        left = boxes[index]
        right = boxes[index + 1]
        if right[0][0] - left[0][0] < 15:
            if abs(left[1][0] - right[1][0]) < 15:
                multi.append(left)
                flag = 1
            else:
                single.append(left)
        else:
            if flag == 0:
                single.append(left)
        index += 1
    left = 0
    right = 0
    #处理多次出现的方框
    result1 = []
    while right < len(multi):
        if right == len(multi) - 1:
            y1s = [multi[k][0][1] for k in range(left, right + 1)]
            y2s = [multi[k][2][1] for k in range(left, right + 1)]
            x1s = [multi[k][0][0] for k in range(left, right + 1)]
            x2s = [multi[k][1][0] for k in range(left, right + 1)]
            y1 = min(y1s)
            y2 = max(y2s)
            x1 = max(x1s)
            x2 = min(x2s)
            result1.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            right += 1
        else:
            if multi[right + 1][0][0] - multi[right][0][0] < 20 and multi[right + 1][1][0] - multi[right][1][0] < 20:
                right += 1
            else:
                y1s = [multi[k][0][1] for k in range(left, right + 1)]
                y2s = [multi[k][2][1] for k in range(left, right + 1)]
                x1s = [multi[k][0][0] for k in range(left, right + 1)]
                x2s = [multi[k][1][0] for k in range(left, right + 1)]
                y1 = max(y1s)
                y2 = min(y2s)
                x1 = max(x1s)
                x2 = min(x2s)
                # y1 = min(y1s)
                # y2 = max(y2s)
                # x1 = min(x1s)
                # x2 = max(x2s)
                result1.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                left = right + 1
                right = left

    #删除逻辑错误的框
    result1.sort(key=lambda x: x[0][0], reverse=False)
    print(result1)
    problems = []
    for m in range(len(result1) - 1):
        left = result1[m]
        right = result1[m + 1]
        if (left[0][0] <= right[0][0] and left[1][0] >= right[1][0] and left[0][1] <= right[0][1] and left[2][1] >= right[2][1]) or \
                right[0][0] <= left[0][0] and right[1][0] >= left[1][0] and right[0][1] <= left[0][1] and right[2][1] >= left[2][1]:
            problems.append(m)
    print(problems)
    for n in range(len(problems) - 1, -1, -1):
        result1.pop(problems[n])
    #画出多次出现的边框
    pre = []
    for k in result1:
        x1 = k[0][0]
        x2 = k[1][0]
        y1 = k[0][1]
        y2 = k[2][1]
        lst = []
        for m in range(x1, x2):
            if m > 874:
                break
            for n in range(y1, y2):
                lst.append(img[n, m])
        x = np.array(lst)
        if np.mean(x) < 136:
            continue
        box = np.array(k, dtype=int)
        pre.append(box)
        cv.drawContours(img2, [box.astype(int)], 0, (255, 255, 255), 1)
        Xs = [k[0] for k in box]
        Ys = [k[1] for k in box]
        x1 = abs(min(Xs))
        x2 = abs(max(Xs))
        y1 = abs(min(Ys))
        y2 = abs(max(Ys))
        hight = y2 - y1
        width = x2 - x1
        save_path = '../res1/' + str(count) + '.jpg'
        crop_img = img2[y1 : y1 + hight, x1 : x1 + width]
        cv.imwrite(save_path, crop_img)
        count += 1
    cv.imshow('original-' + str(j), img2)
    cv.waitKey()
    return pre, count

def DealWiFiWithOutDraw(j):
    img_path = '../recombination/' + str(j) + '.jpg'
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #平滑滤波
    gray = cv.blur(gray, (5, 5))#均值滤波
    gray = cv.fastNlMeansDenoising(gray,None, 10, 7, 21)
    gray = cv.GaussianBlur(gray, (5, 5), 0)#高斯滤波
    boxes = []
    for threshold in range(135, 175):
        #二值化
        ret, mat = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
        #找出二值图中的外边界
        contours, hierarchy = cv.findContours(mat, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = box.astype(int)
            if abs((box[1][0] - box[0][0]) - (box[3][0] - box[2][0])) > 20:
                continue
            Xs = [k[0] for k in box]
            Ys = [k[1] for k in box]
            x1 = abs(min(Xs))
            x2 = abs(max(Xs))
            y1 = abs(min(Ys))
            y2 = abs(max(Ys))
            hight = y2 - y1
            width = x2 - x1
            if width == 0 or hight == 0:
                continue
            #对外边框进行筛选，以得到可能是WiFi的边框
            if (hight >= 90 and hight <= 300) or (150 <= width and width <= 250):
                #cv.drawContours(img, [box.astype(int)], 0, (255, 255, 255), 1)
                boxes.append([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            else:
                continue
    #外边框排序
    boxes.sort(key = lambda x:x[0][0], reverse=False)
    # print(boxes)
    index = 0
    single = []
    flag = 0
    multi = []
    #分离多次出现和单独出现的边框
    while index < len(boxes) - 1:
        if index == len(boxes) - 2:
            left = boxes[index]
            right = boxes[index + 1]
            if right[0][0] - left[0][0] < 20:
                if abs(left[1][0] - right[1][0]) < 20:
                    multi.append(left)
                    multi.append(right)
                else:
                    single.append(left)
            else:
                single.append(left)
        left = boxes[index]
        right = boxes[index + 1]
        if right[0][0] - left[0][0] < 15:
            if abs(left[1][0] - right[1][0]) < 15:
                multi.append(left)
                flag = 1
            else:
                single.append(left)
        else:
            if flag == 0:
                single.append(left)
        index += 1
    left = 0
    right = 0
    #处理多次出现的方框
    result1 = []
    while right < len(multi):
        if right == len(multi) - 1:
            y1s = [multi[k][0][1] for k in range(left, right + 1)]
            y2s = [multi[k][2][1] for k in range(left, right + 1)]
            x1s = [multi[k][0][0] for k in range(left, right + 1)]
            x2s = [multi[k][1][0] for k in range(left, right + 1)]
            y1 = min(y1s)
            y2 = max(y2s)
            x1 = max(x1s)
            x2 = min(x2s)
            result1.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            right += 1
        else:
            if multi[right + 1][0][0] - multi[right][0][0] < 15 or multi[right + 1][1][0] - multi[right][1][0] < 10:
                right += 1
            else:
                y1s = [multi[k][0][1] for k in range(left, right + 1)]
                y2s = [multi[k][2][1] for k in range(left, right + 1)]
                x1s = [multi[k][0][0] for k in range(left, right + 1)]
                x2s = [multi[k][1][0] for k in range(left, right + 1)]
                y1 = max(y1s)
                y2 = min(y2s)
                x1 = max(x1s)
                x2 = min(x2s)
                result1.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                left = right + 1
                right = left
    #画出多次出现的边框
    pre = []
    for k in result1:
        x1 = k[0][0]
        x2 = k[1][0]
        y1 = k[0][1]
        y2 = k[2][1]
        lst = []
        for m in range(x1, x2):
            if m > 874:
                break
            for n in range(y1, y2):
                lst.append(img[n, m])
        x = np.array(lst)
        if np.mean(x) < 136:
            continue
        box = np.array(k, dtype=int)
        pre.append(box)
        cv.drawContours(img, [box.astype(int)], 0, (255, 255, 255), 1)
        # Xs = [k[0] for k in box]
        # Ys = [k[1] for k in box]
        # x1 = abs(min(Xs))
        # x2 = abs(max(Xs))
        # y1 = abs(min(Ys))
        # y2 = abs(max(Ys))
        # hight = y2 - y1
        # width = x2 - x1
        # save_path = '../result/' + str(count) + '.jpg'
        # crop_img = img[y1 : y1 + hight, x1 : x1 + width]
        # cv.imwrite(save_path, crop_img)
    # cv.imshow('original1-' + str(j), img)
    # cv.waitKey()
    return pre

def DealBlueTooth(count, pre, j):
    #处理蓝牙和私有协议
    img_path = '../recombination/' + str(j) + '.jpg'
    img = cv.imread(img_path)
    img1 = img.copy()
    img2 = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 平滑滤波
    gray = cv.blur(gray, (5, 5))  # 均值滤波
    gray = cv.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = cv.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波
    # 置黑
    for new_box in pre:
        x1 = new_box[0][0]
        x2 = new_box[1][0]
        y1 = new_box[0][1]
        y2 = new_box[2][1]
        for m in range(x1 - 10, x2 + 10):
            if m > 874:
                break
            for n in range(y1 - 10, y2 + 10):
                gray[n, m] = 0
    #重新二值化并遍历阈值画白框
    for threshold in range(80, 190):
        # 二值化
        ret, mat = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
        # 找出二值图中的外边界
        contours, hierarchy = cv.findContours(mat, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = box.astype(int)
            if abs((box[1][0] - box[0][0]) - (box[3][0] - box[2][0])) > 20:
                continue
            cv.drawContours(img, [box.astype(int)], 0, (255, 255, 255), 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.blur(img, (3, 3))  # 均值滤波
    img = cv.fastNlMeansDenoising(img, None, 10, 7, 21)
    img = cv.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    ret, mat = cv.threshold(img, 250, 255, cv.THRESH_BINARY)
    new_contours, new_hierarchy = cv.findContours(mat, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    new_boxes = []
    for cnt in new_contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = box.astype(int)
        Xs = [k[0] for k in box]
        Ys = [k[1] for k in box]
        x1 = abs(min(Xs))
        x2 = abs(max(Xs))
        y1 = abs(min(Ys))
        y2 = abs(max(Ys))
        hight = y2 - y1
        width = x2 - x1
        if width == 0 or hight == 0:
            continue
        if hight / width > 10:
            continue
        if (hight <= 70) and (width <= 15):
            new_boxes.append([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        else:
            continue
        #cv.drawContours(img1, [box.astype(int)], 0, (255, 255, 255), 1)
    # 对新的边框进行筛选排序
    new_boxes.sort(key=lambda x: x[0][0], reverse=False)
    left = right = 0
    indexes = []
    while right < len(new_boxes) - 1:
        if left == right:
            indexes.append([left, right])
            if new_boxes[right + 1][0][0] - new_boxes[right][0][0] < 10:
                right += 1
            else:
                right += 1
                left = right
            continue
        if right == len(new_boxes) - 2:
            if new_boxes[right + 1][0][0] - new_boxes[right][0][0] < 10:
                indexes[-1][1] = right + 1
            else:
                indexes[-1][1] = right
                indexes.append([right + 1, right + 1])
            break
        if new_boxes[right + 1][0][0] - new_boxes[right][0][0] < 10:
            right += 1
        else:
            indexes[-1][1] = right
            right += 1
            left = right
    result2 = []
    for k in indexes:
        left = k[0]
        right = k[1]
        x1s = [new_boxes[i][0][0] for i in range(left, right + 1)]
        x2s = [new_boxes[i][1][0] for i in range(left, right + 1)]
        y1s = [new_boxes[i][0][1] for i in range(left, right + 1)]
        y2s = [new_boxes[i][2][1] for i in range(left, right + 1)]
        y1 = min(y1s)
        y2 = max(y2s)
        x1 = min(x1s)
        x2 = max(x2s)
        result2.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    for k in result2:
        x1 = k[0][0]
        x2 = k[1][0]
        y1 = k[0][1]
        y2 = k[2][1]
        box = np.array(k, dtype=int)
        #cv.drawContours(img1, [box.astype(int)], 0, (255, 255, 255), 1)
    result2.sort(key=lambda x: x[0][0], reverse=False)
    result3 = []
    q = 0
    while q < len(result2) - 1:
        if result2[q + 1][0][0] - result2[q][1][0] < 20 and (abs(result2[q + 1][0][1] - result2[q][0][1]) < 70 or abs(result2[q + 1][2][1] - result2[q][2][1]) < 70):
            y1 = min(result2[q][0][1], result2[q + 1][0][1])
            y2 = max(result2[q][2][1], result2[q + 1][2][1])
            x1 = result2[q][0][0]
            x2 = result2[q + 1][1][0]
            result3.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            q += 2
        else:
            result3.append(result2[q])
            if q == len(result2) - 2:
                result3.append(result2[q + 1])
            q += 1
    for k in result3:
        box = np.array(k, dtype=int)
        cv.drawContours(img2, [box.astype(int)], 0, (255, 255, 255), 1)
        Xs = [k[0] for k in box]
        Ys = [k[1] for k in box]
        x1 = abs(min(Xs))
        x2 = abs(max(Xs))
        y1 = abs(min(Ys))
        y2 = abs(max(Ys))
        hight = y2 - y1
        width = x2 - x1
        # save_path = '../test/' + str(count) + '.jpg'
        # crop_img = img2[y1 : y1 + hight, x1 : x1 + width]
        # cv.imwrite(save_path, crop_img)
        # count += 1
    # cv.imshow('original2-' + str(j), img2)
    # cv.waitKey()
    return count

if __name__ == '__main__':
    count = 1
    for j in range(2, 101):
        pre, count = DealWiFi(count, j)
        print(count)
        # pre1 = DealWiFiWithOutDraw(j)
        # count2 = DealBlueTooth(count1, pre1, j)
