import cv2 as cv
import numpy as np

def cv_show(name,img):
	cv.imshow(name, img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def sort_contours(cnts):
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    print(boundingBoxes)
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0], reverse=False))

    return cnts, boundingBoxes

def compare(digits,last):
    score=[]
    out = []
    for (i,digitsROI) in digits.items():
        result =cv.matchTemplate(last,digitsROI,cv.TM_CCOEFF_NORMED)
        (min,max,minlo,maxloc)=cv.minMaxLoc(result)
        score.append(max)
    ot=out.append(str(np.argmax(score)))
    print(ot)




def credit_card(image, img):
    grayt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, bianryt = cv.threshold(grayt, 127, 255, cv.THRESH_BINARY_INV)
    # cv.imshow('binaryt',bianryt)
    containt, hierachy = cv.findContours(bianryt.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(image, containt, -1, (0, 0, 255), 2)
    # cv_show('binarytcopy',image)
    # for i in range (10):
    #     cv.drawContours(image, containt[i], -1, (0,0,255), 2)
    #     cv_show('binarytcopy', image)


    print(np.array(containt).shape)
    # print(containt)
    refcnts = sort_contours(containt)[0]

    for i in range (10):
        cv.drawContours(image, refcnts[i], -1, (0,0,255), 2)
        cv_show('binarytcopy', image)

    digits = {}
    boundt = [cv.boundingRect(i) for i in containt]
    boundt = sorted(boundt, key=lambda x: x[0])
    print('1',boundt)

    for (i, c) in enumerate(refcnts):
        (x, y, w, h) = cv.boundingRect(c)
        roi = bianryt[y:y + h, x:x + w]
        roi = cv.resize(roi, (57, 88))
        digits[i] = roi
    print('digist',digits)

    retkernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 2))
    sqkernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    graym = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv_show('graym',graym)
    tophat = cv.morphologyEx(graym, cv.MORPH_TOPHAT, retkernel)
    # cv.imshow('tophat',tophat)
    gradx = cv.Sobel(tophat, cv.CV_32F, 1, 0, -1)
    imgx = cv.convertScaleAbs(gradx)
    # cv_show('imgx',imgx)
    # print('gradx',gradx)
    # print('gradx.shape', gradx.shape)
    imgx = cv.dilate(imgx, retkernel, iterations=3)
    # imgxx=cv.morphologyEx(imgx,cv.MORPH_OPEN,retkernel,iterations=1)
    # cv.imshow('imgxx',imgx)
    ret, thresholdx = cv.threshold(imgx, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv_show('thresholdx', thresholdx)

    containtx, hierachy = cv.findContours(thresholdx, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, containtx, -1, (0, 0, 255), 2)
    cv_show('imagecopy', img)
    print(np.array(containtx).shape)

    boundx = [cv.boundingRect(i) for i in containtx]
    # print(boundx)

    roix = []
    groupOutput = []

    for i in range(len(boundx)):
        if (110 < boundx[i][2] < 130) and (20 < boundx[i][3] < 40):
            roix.append(boundx[i])
    roix = sorted(roix, key=lambda x: x[0])
    print(roix)

    for (i,(xx,xy,xw,xh)) in enumerate(roix):

        showx=graym[xy:xy+xh,xx:xx+xw]
        cv_show('showx',showx)
        ret,showbinary=cv.threshold(showx,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
        cv_show('showbinary',showbinary)
        containtbinary,hierachy=cv.findContours(showbinary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        showx=cv.cvtColor(showx,cv.COLOR_GRAY2BGR)
        boundbinary = [cv.boundingRect(i) for i in containtbinary]
        boundbinary = sorted(boundbinary, key=lambda x: x[0])

        #
        # for i in range(len(boundbinary)):
        #     cv.drawContours(showbinary,boundbinary[i],-1,(0,0,255),2)
        #     cv.imshow('showbinaryy',showbinary)
        #


        # print(boundbinary)
        for i in range(len(boundbinary)):
            gh=boundbinary[i][3]
            gx=boundbinary[i][0]
            gw=boundbinary[i][2]
            last=showbinary[0:gh,gx:gx+gw]
            last = cv.resize(last, (57, 88))
            cv_show('last',last)

            # compare(digits,last)
            scores = []
            for (digit, digitROI) in digits.items():
                # 模板匹配
                result = cv.matchTemplate(last, digitROI,
                                           cv.TM_CCOEFF)
                (a, score, _, _) = cv.minMaxLoc(result)
                scores.append(score)
                print('scores',scores)

            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))
            print('groupOutput',groupOutput)
    a = ''.join(groupOutput)
    print('a=',a)









        # for i in range(len(roix)):
        #     last=showx[0:roix[i][3],(i/4)*roix[i][2]:((i+1)/4)*roix[i][2]]
        #     cv_show('last',last)
        # for i in range (len(roix)):
        #     gh=roix[i][3]
        #     gx=int((i/4)*roix[i][2])
        #     gw=int(((i+1)/4)*roix[i][2])
        #     print(0,gh,gx,gw)
        #     last=showx[0:gh,gx:gw]
        #     cv_show('last',last)





t = cv.imread(r'image\ocr_a_reference.png')
# cv.imshow('t',t)
m = cv.imread(r'image\credit_card_01.png')
# cv.imshow('m',m)
# m=cv.resize(m,(300,189))
credit_card(t, m)
cv.waitKey(0)
cv.destroyAllWindows()
