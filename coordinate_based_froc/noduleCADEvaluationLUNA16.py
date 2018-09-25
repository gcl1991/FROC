# coding:utf-8
import os
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogFormatter,StrMethodFormatter,FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from NoduleFinding import NoduleFinding

from tools import csvTools

# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 512 # Maximum value of x-axis of FROC curve
bLogPlot = True

def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_index_im   = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

    return candidates

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        # 原程序
        # sens_lb[i] = vec[math.floor(Pz * len(vec))]
        # sens_up[i] = vec[math.floor((1.0 - Pz) * len(vec))]
        # 改写后
        sens_lb[i] = vec[int(np.floor(Pz*len(vec)))]
        sens_up[i] = vec[int(np.floor((1.0-Pz)*len(vec)))]

    return sens_mean,sens_lb,sens_up

def computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,FROCImList,excludeList,numberOfBootstrapSamples=1000, confidence = 0.95):

    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
    
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)
    
    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:,i:i+1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid],candidate),axis = 1)

    for i in range(numberOfBootstrapSamples):
        print 'computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict,FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
    
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # compute mean and CI
    sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)

    return all_fps, sens_mean, sens_lb, sens_up

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): #  Handle border case when there are no false positives and ROC analysis give nan values.
      print "WARNING, this system has no false positives.."
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False,numberOfBootstrapSamples=1000,confidence = 0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''
    # evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
    #             os.path.splitext(os.path.basename(results_filename))[0],
    #             # 去掉尾缀和路径后的文件名主干 ./submission/sampleSubmission.csv 变为sampleSubmission
    #             maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
    #             numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)
    # bPerformBootstrapping = True  # Bootstrap是用小样本估计总体值的一种非参数方法，有放回采样
    # bNumberOfBootstrapSamples = 1000  # 采样数量
    # bOtherNodulesAsIrrelevant = True 其他不相关结节
    # bConfidence = 0.95


    nodOutputfile = open(os.path.join(outputDir,'CADAnalysis.txt'),'w')
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    results = csvTools.readCSV(results_filename)

    allCandsCAD = {}
    # 将预测结果中病人：所有结节（字典） 存储成字典形式 ，结节字典key为编号0 1....n，结节数量小于等于maxNumberOfCADMarks，超过则按概率从大到取
    for seriesuid in seriesUIDs:
        
        # collect candidates from result file
        nodules = {}
        header = results[0]
        
        i = 0
        # 获得每个病人的结节 并编号0 1 ... n 存储到字典
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:  #  nodules.keys()函数以列表返回一个字典所有的键
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.iteritems():  # 返回可遍历的(键, 值) 元组数组迭代器
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.iteritems():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2  # maxNumberOfCADMarks个结节
        
        print 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules  # 预测文件中的结节字典 病人id:所有结节，所有结节以0 1 ...n 和结节信息的字典形式存储
    
    # open output files
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')
    
    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0  # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases 遍历每个病人
    for seriesuid in seriesUIDs:
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]  # 获取当前病人预测数据的字典
            # 当前病人所有结节（字典） ，结节字典key为编号0 1....n，结节数量小于等于maxNumberOfCADMarks，超过则按概率从大到取
            # key = 0 1 ...n; content 结节类对象 x y z p candidateID=key其他默认
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys())  # 当前病人预测结节的总数

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]
            # 获取当前病人结节数据，结节（候选+额外的）以列表形式存储，其中每个结节以结节类对象形式保存
            # 结节列表x,y,z,d,state=Included or Excluded，其他为默认
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations 外循环：遍历当前病人的每一个实际结节
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1  # 每个人实际Included结节的总数

            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm)
            if diameter < 0.0:
              diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)

            found = False
            noduleMatches = []
            for key, candidate in candidates.iteritems():  # 内循环：遍历当前病人预测的结节字典，key为编号0 1....n
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)
                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
                if dist < radiusSquared:
                    if (noduleAnnot.state == "Included"):
                        found = True  # 在圆内，标记为预测到了，此时没考虑预测的概率
                        noduleMatches.append(candidate)  # 将找到的结节添加到列表
                        if key not in candidates2.keys():  # 重复预测警告
                            print "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), seriesuid, str(noduleAnnot.id))
                        else:
                            del candidates2[key]
                    elif (noduleAnnot.state == "Excluded"):  # an excluded nodule
                        if bOtherNodulesAsIrrelevant: #    delete marks on excluded nodules so they don't count as false positives
                            if key in candidates2.keys():
                                irrelevantCandidates += 1  # 无关结节，不算误报
                                # 不算误报的结节列表
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
                                del candidates2[key]
            if len(noduleMatches) > 1:  # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)  # 忽略的结节数量？？？
            if noduleAnnot.state == "Included":  # 如果是候选结节
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    # 获得结节预测概率最大值
                    for idx in range(len(noduleMatches)):
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROCGTList.append(1.0)   # 实际标签 追加1
                    FROCProbList.append(float(maxProb))  # 预测概率最大值 追加
                    FPDivisorList.append(seriesuid)  # 病人UID
                    excludeList.append(False)  # 排除开关:不排除
                    # FROC 结节 seriesuid：病人UID noduleAnnot.id:None  candidate.id:None candidate.CADprobability:列表中最后一个结节的概率
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    # FROC图
                    candTPs += 1  # 候选真阳性+1
                # 未检测到
                else:
                    candFNs += 1  # FN + 1 假阴性
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROCGTList.append(1.0)  # 实际标签 追加1
                    FROCProbList.append(minProbValue)  # 追加最小概率
                    FPDivisorList.append(seriesuid)  # 病人UID
                    excludeList.append(True)  # 排除开关：排除
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), int(-1), "NA"))
                    nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(-1)))
                    # nodNoCandFile:nodulesWithoutCandidate.txt 未检测到存储1的文件
        # add all false positives to the vectors
        for key, candidate3 in candidates2.iteritems():  # candidates2此时剩下的是误报的，不在圆内
            candFPs += 1
            FROCGTList.append(0.0)  # 实际标签 追加0
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id), float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")  # 错误报告
    # 统计信息
    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write("    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
    nodOutputfile.write("    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)
    
    if performBootstrapping:  # True
        fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,seriesUIDs,excludeList,
                                                                  numberOfBootstrapSamples=numberOfBootstrapSamples, confidence = confidence)
        
    # Write FROC curve
    with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
    
    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)  # FROC横坐标范围
    
    sens_itp = np.interp(fps_itp, fps, sens)  # FROC纵坐标
    
    if performBootstrapping: # True
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))
        
        if bLogPlot:
            plt.xscale('log', basex=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)
    
def getNodule(annotation, header, state = ""):
    # 创建结节类对象，并存储state信息 x,y,z,d,p,state
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]
    
    if diameter_mm_label in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]
    
    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]
    
    if not state == "":
        nodule.state = state

    return nodule

def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    # 将每个人的结节列表(x,y,z,d,state)和病人名字，以字典形式保存，返回这个字典，并统计Included结节数量和Included+excluded结节数量
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        print 'adding nodule annotations: ' + seriesuid
        
        nodules = []
        numberOfIncludedNodules = 0
        
        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state = "Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1
        
        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state = "Excluded")
                nodules.append(nodule)
            
        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)
    
    print 'Total number of included nodule annotations: ' + str(noduleCount)
    print 'Total number of nodule annotations: ' + str(noduleCountTotal)
    return allNodules
    
    
def collect(annotations_filename,annotations_excluded_filename,seriesuids_filename):
    '''
    返回 病人-结节_字典 和 病人UID_list type:tuple
    :param annotations_filename:
    :param annotations_excluded_filename:
    :param seriesuids_filename:
    :return:
    '''
    annotations = csvTools.readCSV(annotations_filename)  # csv 转 list
    annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
    seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)
    
    seriesUIDs = []
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])

    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)  # content 病人：结节类对象 type dict
    
    return (allNodules, seriesUIDs)
    
    
def noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''
    
    print annotations_filename
    
    (allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
    # allNodules 存储所有病人-结节的字典，seriesUIDs所有病人id的列表
    
    evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
                os.path.splitext(os.path.basename(results_filename))[0],  # 去掉尾缀和路径后的文件名主干 ./submission/sampleSubmission.csv 变为sampleSubmission
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)
    # bPerformBootstrapping = True  # Bootstrap是用小样本估计总体值的一种非参数方法，有放回采样
    # bNumberOfBootstrapSamples = 1000  # 采样数量
    # bOtherNodulesAsIrrelevant = True 其他不相关结节
    # bConfidence = 0.95

if __name__ == '__main__':
    # 原文
    # annotations_filename          = sys.argv[1]  # ./annotations/annotations.csv
    # annotations_excluded_filename = sys.argv[2]  # ./annotations/annotations_excluded.csv
    # seriesuids_filename           = sys.argv[3]  # ./annotations/seriesuids.csv
    # results_filename              = sys.argv[4]  # ./submission/sampleSubmission.csv
    # outputDir                     = sys.argv[5]  # ./result
    # 修改
    annotations_filename          = './annotations/annotations.csv'
    annotations_excluded_filename = './annotations/annotations_excluded.csv'
    seriesuids_filename           = './annotations/seriesuids.csv'
    results_filename              = './submission/sampleSubmission.csv'
    outputDir                     = './result'

    # execute only if run as a script
    noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir)
    print "Finished!"
